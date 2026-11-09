#include "operations.hpp"
#include <Robotiq/gripper/modbus_client.hpp>
#include <Robotiq/gripper/connection_config.hpp>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
  try {
    if (argc != 10) throw std::invalid_argument(
      "Usage: robotiq_sdk_command status|move|activate PORT BAUD SLAVE SERIAL_TIMEOUT ACTION_TIMEOUT SPEED WIDTH FORCE");
    const std::string operation = argv[1];
    const int baud = std::stoi(argv[3]), slave = std::stoi(argv[4]);
    const double serialTimeout = std::stod(argv[5]), actionTimeout = std::stod(argv[6]);
    const int speed = std::stoi(argv[7]), force = std::stoi(argv[9]);
    const double width = std::stod(argv[8]);
    if ((operation != "status" && operation != "move" && operation != "command" && operation != "activate") ||
        !*argv[2] || baud <= 0 || slave < 1 || slave > 247 || speed < 0 || speed > 255 ||
        !std::isfinite(serialTimeout) || serialTimeout < 0.001 || serialTimeout > 3600 ||
        !std::isfinite(actionTimeout) || actionTimeout <= 0)
      throw std::invalid_argument("Invalid SDK command or connection parameters");
    if (operation == "move" || operation == "command") {
      harvest::positionForWidth(width);
      if (force < 0 || force > 255) throw std::invalid_argument("force must be raw 0..255");
    }
    Robotiq::ConnectionConfig config;
    config.serial.port = argv[2];
    config.serial.baudrate = baud;
    config.serial.timeout = std::chrono::milliseconds(static_cast<long long>(serialTimeout * 1000));
    config.serial.latencyTimerMs = 0; // Do not modify host sysfs settings.
    config.modbusSlaveAddress = slave;
    // Official public transaction API: deliberately no background exchange,
    // so diagnostic status queries cannot write/reset the command block.
    Robotiq::GripperModbusClient client(config);
    Robotiq::GripperStatus status;
    if (operation == "status") status = harvest::statusBeforeMove(client, actionTimeout);
    else if (operation == "command") {
      // Send once without waiting for finger travel; Python continues polling real feedback.
      (void)harvest::statusBeforeMove(client, actionTimeout);
      Robotiq::GripperCommand command{};
      command.action.set(Robotiq::ActionRequestBit::Activate);
      command.action.set(Robotiq::ActionRequestBit::GoTo);
      command.positionRequest = harvest::positionForWidth(width);
      command.speed = speed;
      command.force = force;
      client.writeCommand(command);
      status = client.readStatus();
    }
    else if (operation == "move") status = harvest::move(client, width, force, speed, actionTimeout);
    else status = harvest::activate(client, actionTimeout);
    std::cout << "{\"flags\":" << int(status.gripperStatus.raw())
      << ",\"fault\":" << harvest::fault(status)
      << ",\"requested\":" << int(status.positionRequestEcho)
      << ",\"position\":" << int(status.position)
      << ",\"current\":" << int(status.current) << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
