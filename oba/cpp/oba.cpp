#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

struct Event {
    long timestamp;
    std::string side;
    std::string action;
    int id;
    int price;
    int quantity;
};

int main() {
    std::vector<Event> events;
    std::ifstream inputFile("../../data/res_20190610.csv");

    if (!inputFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    std::string line;
    int line_number = 0;

    while (std::getline(inputFile, line)) {

        // for some reason it doesn't work to check whether its zero
        // and putting the incrementation of line_number at the end of 
        // the while loop
        line_number++;
        if (line_number == 1){
            continue;
        }

        std::istringstream iss(line);
        std::string timestamp, side, action, id, price, quantity;

        try {
        if (std::getline(iss, timestamp, ',') &&
            std::getline(iss, side, ',') &&
            std::getline(iss, action, ',') &&
            std::getline(iss, id, ',') &&
            std::getline(iss, price, ',') &&
            std::getline(iss, quantity, ',')) {

            Event event{
                std::stol(timestamp),
                side,
                action,
                std::stoi(id),
                std::stoi(price),
                std::stoi(quantity)
            };

            events.push_back(event);
        } else {
            std::cerr << "Error parsing line: " << line << std::endl;
        }
        }
         catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << " in line: " << line << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range error: " << e.what() << " in line: " << line << std::endl;
        }
    }

    inputFile.close();


    for (const auto& event : events) {
        std::cout << "Timestamp: " << event.timestamp << ", "
                  << "Side: " << event.side << ", "
                  << "Action: " << event.action << ", "
                  << "ID: " << event.id << ", "
                  << "Price: " << event.price << ", "
                  << "Quantity: " << event.quantity << std::endl;
    }
}
