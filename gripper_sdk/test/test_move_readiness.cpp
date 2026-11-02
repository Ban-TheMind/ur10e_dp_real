#include "operations.hpp"
#include <deque>
#include <iostream>
#include <utility>
#include <vector>

struct FakeClient {
  std::deque<Robotiq::GripperStatus> statuses;
  std::vector<Robotiq::GripperCommand> commands;
  int reads = 0;
  int readsBeforeWrite = -1;
  bool repeatLast = false;
  bool failWrite = false;
  explicit FakeClient(std::deque<Robotiq::GripperStatus> replies)
    : statuses(std::move(replies)) {}
  Robotiq::GripperStatus readStatus() {
    ++reads;
    if (statuses.empty()) throw std::runtime_error("simulated transport failure");
    auto s = statuses.front();
    if (!repeatLast || statuses.size() > 1) statuses.pop_front();
    return s;
  }
  void writeCommand(const Robotiq::GripperCommand& c) {
    readsBeforeWrite = reads;
    commands.push_back(c);
    if (failWrite) throw std::runtime_error("simulated write timeout");
  }
};
Robotiq::GripperStatus state(uint8_t flags=0xf9, uint8_t fault=0, uint8_t echo=105) {
  Robotiq::GripperStatus s;
  s.gripperStatus = Robotiq::GripperStatusFlags::fromRaw(flags);
  s.faultStatus = Robotiq::FaultStatus::fromRaw(fault);
  s.positionRequestEcho = echo;
  s.position = echo;
  return s;
}
void check(bool condition) {
  if (!condition) throw std::runtime_error("readiness regression assertion failed");
}
template<class F> void mustFailWith(F f, const std::string& message) {
  try { f(); }
  catch (const std::exception& error) {
    check(std::string(error.what()).find(message) != std::string::npos);
    return;
  }
  throw std::runtime_error("expected failure: " + message);
}
int main() {
  try {
    FakeClient ready{{state(), state()}};
    check(harvest::move(ready,50,0,16,1).position == 105);
    check(ready.readsBeforeWrite == 1 && ready.commands.size() == 1);

    FakeClient recovered{{state(0xf9,9), state(), state()}};
    check(harvest::move(recovered,50,0,16,1).position == 105);
    check(recovered.readsBeforeWrite == 2 && recovered.commands.size() == 1);
    check(recovered.commands[0].positionRequest == 105);
    check(recovered.commands[0].speed == 16 && recovered.commands[0].force == 0);
    check(recovered.commands[0].action.value() == 9); // No reset/activation-only write.

    FakeClient delayed{{state(0xf9,9),state(0xf9,9),state(),state()}};
    harvest::move(delayed,50,0,16,1);
    check(delayed.readsBeforeWrite == 3 && delayed.commands.size() == 1);

    FakeClient persistent{{state(0xf9,9)}};
    persistent.repeatLast = true;
    mustFailWith([&]{harvest::move(persistent,50,0,16,0.12);}, "fault code 9");
    check(persistent.commands.empty() && persistent.reads <= 11);

    FakeClient bounded{{state(0xf9,9)}};
    bounded.repeatLast = true;
    const auto started = std::chrono::steady_clock::now();
    mustFailWith([&]{harvest::move(bounded,50,0,16,10);}, "no command sent");
    check(bounded.commands.empty() && bounded.reads <= 11);
    check(std::chrono::steady_clock::now() - started < std::chrono::seconds(2));

    FakeClient faulted{{state(0xf9,14)}};
    mustFailWith([&]{harvest::move(faulted,50,0,16,1);}, "fault code 14");
    check(faulted.commands.empty() && faulted.reads == 1);

    FakeClient newFault{{state(0xf9,9),state(0xf9,8)}};
    mustFailWith([&]{harvest::move(newFault,50,0,16,1);}, "fault code 8");
    check(newFault.commands.empty() && newFault.reads == 2);

    FakeClient unready{{state(0xf9,9),state(0)}};
    mustFailWith([&]{harvest::move(unready,50,0,16,1);}, "not activated/ready");
    check(unready.commands.empty());

    FakeClient transport{{state(0xf9,9)}};
    mustFailWith([&]{harvest::move(transport,50,0,16,1);}, "transport failure");
    check(transport.commands.empty());

    FakeClient afterWrite{{state(),state(0xf9,9)}};
    mustFailWith([&]{harvest::move(afterWrite,50,0,16,1);}, "fault code 9");
    check(afterWrite.commands.size() == 1 && afterWrite.reads == 2);

    FakeClient writeTimeout{{state()}};
    writeTimeout.failWrite = true;
    mustFailWith([&]{harvest::move(writeTimeout,50,0,16,1);}, "write timeout");
    check(writeTimeout.commands.size() == 1 && writeTimeout.reads == 1);

    std::cout << "11 move-readiness regression scenarios passed; no hardware contacted\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
