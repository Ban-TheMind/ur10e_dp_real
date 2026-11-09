#pragma once
#include <Robotiq/gripper/command.hpp>
#include <Robotiq/gripper/status.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>

namespace harvest {
using Robotiq::GripperStatus;
inline bool ready(const GripperStatus& s) {
  return s.gripperStatus.activated() &&
    s.gripperStatus.activationState() == Robotiq::ActivationState::Complete;
}
inline int fault(const GripperStatus& s) {
  return static_cast<int>(s.faultStatus.gripperFault());
}
inline int positionForWidth(double width) {
  if (!std::isfinite(width) || width < 0 || width > 85)
    throw std::invalid_argument("width must be 0..85 mm");
  // Same nominal mapping as the service, not a calibrated finger gap.
  return static_cast<int>(std::nearbyint((85 - width) * 255 / 85));
}
template<class Client, class Predicate>
GripperStatus wait(Client& client, Predicate predicate, double timeout, bool activating=false) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout);
  while (std::chrono::steady_clock::now() < deadline) {
    auto s = client.readStatus();
    const int code = fault(s);
    if (code && !(activating && (code == 5 || code == 7)))
      throw std::runtime_error("Gripper fault code " + std::to_string(code));
    if (predicate(s)) return s;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  throw std::runtime_error("Gripper action timed out; physical state unknown");
}
// Recover only an idle communication timeout, before writing any command.
// Bound both the polling time and read count; transport errors propagate immediately.
template<class Client>
GripperStatus statusBeforeMove(Client& client, double timeout) {
  using Clock = std::chrono::steady_clock;
  const auto budget = std::chrono::duration_cast<Clock::duration>(
    std::chrono::duration<double>(std::min(timeout, 0.5)));
  const auto deadline = Clock::now() + budget;
  auto initial = client.readStatus();
  int retries = 0;
  while (initial.faultStatus.gripperFault() == Robotiq::GripperFault::NoCommunication) {
    if (retries >= 10 || Clock::now() >= deadline)
      throw std::runtime_error("Gripper communication timeout (fault code 9) did not clear before motion; no command sent");
    std::this_thread::sleep_until(std::min(deadline, Clock::now() + std::chrono::milliseconds(50)));
    if (Clock::now() >= deadline)
      throw std::runtime_error("Gripper communication timeout (fault code 9) did not clear before motion; no command sent");
    initial = client.readStatus();
    ++retries;
  }
  if (fault(initial))
    throw std::runtime_error("Gripper fault code " + std::to_string(fault(initial)) +
                             " before motion; no command sent");
  if (!ready(initial))
    throw std::runtime_error("Gripper not activated/ready; explicit reset/activation required; no command sent");
  return initial;
}
template<class Client>
GripperStatus move(Client& client, double width, int force, int speed, double timeout) {
  const auto target = positionForWidth(width);
  if (force < 0 || force > 255 || speed < 0 || speed > 255)
    throw std::invalid_argument("force and speed must be raw 0..255 values");
  (void)statusBeforeMove(client, timeout);
  Robotiq::GripperCommand command{};
  command.action.set(Robotiq::ActionRequestBit::Activate);
  command.action.set(Robotiq::ActionRequestBit::GoTo);
  command.positionRequest = target;
  command.speed = speed;
  command.force = force;
  client.writeCommand(command);
  auto s = wait(client, [target](const auto& status) {
    return ready(status) && status.gripperStatus.goToEnabled() &&
      status.positionRequestEcho == target &&
      status.gripperStatus.objectDetection() != Robotiq::ObjectDetection::Moving;
  }, timeout);
  if (s.gripperStatus.objectDetection() == Robotiq::ObjectDetection::DetectedWhileOpening)
    throw std::runtime_error("Gripper stopped on contact while opening");
  return s;
}
template<class Client>
GripperStatus activate(Client& client, double timeout) {
  Robotiq::GripperCommand command{};
  client.writeCommand(command);
  wait(client, [](const auto& s) { return !s.gripperStatus.activated(); }, timeout, true);
  command.action.set(Robotiq::ActionRequestBit::Activate);
  client.writeCommand(command);
  return wait(client, [](const auto& s) { return ready(s) && !fault(s); }, timeout, true);
}
}
