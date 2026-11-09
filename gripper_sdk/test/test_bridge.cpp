#include "operations.hpp"
#include "test_utils.hpp"
#include <Robotiq/gripper/modbus_client.hpp>
#include <deque>
#include <iostream>
#include <vector>

struct FakeClient {
  std::deque<Robotiq::GripperStatus> statuses;
  std::vector<Robotiq::GripperCommand> commands;
  Robotiq::GripperStatus readStatus() {
    if(statuses.empty()) throw std::runtime_error("no simulated reply");
    auto s = statuses.front(); statuses.pop_front(); return s;
  }
  void writeCommand(const Robotiq::GripperCommand& c) {commands.push_back(c);}
};
Robotiq::GripperStatus state(uint8_t flags=0xf9, uint8_t echo=255, uint8_t fault=0) {
  Robotiq::GripperStatus s;
  s.gripperStatus = Robotiq::GripperStatusFlags::fromRaw(flags);
  s.positionRequestEcho = echo;
  s.faultStatus = Robotiq::FaultStatus::fromRaw(fault);
  return s;
}
void check(bool condition) { if(!condition) throw std::runtime_error("assertion failed"); }
template<class F> void mustFail(F f) {
  bool threw=false; try { f(); } catch(const std::exception&) {threw=true;} check(threw);
}
int main() {
  try {
    // Exercise the real official SDK over its own injectable Serial fixture.
    auto serial = std::make_unique<Robotiq::test::ScriptedSerial>();
    auto* wire = serial.get();
    Robotiq::GripperModbusClient sdk(std::move(serial), 9);
    check(wire->written().empty()); // Constructing the SDK sends no commands.
    wire->preloadRead(Robotiq::test::withCrc({9,3,6,0xf9,0,0,255,250,0}));
    check(sdk.readStatus().position == 250);
    check(wire->written() == Robotiq::test::readHoldingRegistersFrame(9,0x07d0,3));
    auto bad = Robotiq::test::withCrc({9,3,6,0xf9,0,0,255,250,0});
    bad.back() ^= 1;
    wire->preloadRead(bad);
    mustFail([&]{ (void)sdk.readStatus(); });
    check(harvest::positionForWidth(85)==0 && harvest::positionForWidth(0)==255);
    FakeClient moving{{state(), state(0xf9,0), state(0x39), state(0xb9)}};
    check(harvest::move(moving,0,40,64,1).gripperStatus.objectDetection()==Robotiq::ObjectDetection::DetectedWhileClosing);
    check(moving.statuses.empty() && moving.commands.size()==1);
    check(moving.commands[0].positionRequest==255 && moving.commands[0].force==40);
    FakeClient opening{{state(),state(0x79,0)}};
    mustFail([&]{ harvest::move(opening,85,10,64,1); });
    FakeClient unready{{state(0)}};
    mustFail([&]{ harvest::move(unready,0,10,64,1); }); check(unready.commands.empty());
    FakeClient invalid;
    mustFail([&]{ harvest::move(invalid,100,10,64,1); }); check(invalid.commands.empty());
    FakeClient faulty{{state(),state(0xb9,255,14)}};
    mustFail([&]{ harvest::move(faulty,0,10,64,1); });
    FakeClient reset{{state(0), state(0x11), state(0x31)}};
    check(harvest::ready(harvest::activate(reset,1)));
    check(reset.commands.size()==2 && reset.commands[0].action.value()==0 && reset.commands[1].action.value()==1);
    FakeClient timeout;
    mustFail([&]{ harvest::wait(timeout,[](const auto&){return false;},0); });
    std::cout << "Official-SDK bridge policy tests passed\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n'; return 1;}
}
