#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

struct RawUpdate {
    int timestamp;
    std::string side;
    std::string action;
    int id;
    int price;
    int quantity;
};

int main() {
    std::vector<RawUpdate> updates;
    std::ifstream inputFile("../data/test_input.csv");

    if (!inputFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        std::string timestamp, side, action, id, price, quantity;

        if (std::getline(iss, timestamp, ',') &&
            std::getline(iss, side, ',') &&
            std::getline(iss, action, ',') &&
            std::getline(iss, id, ',') &&
            std::getline(iss, price, ',') &&
            std::getline(iss, quantity, ',')) {

            RawUpdate update{
                std::stoi(timestamp),
                side,
                action,
                std::stoi(id),
                std::stoi(price),
                std::stoi(quantity)
            };

            updates.push_back(update);
        } else {
            std::cerr << "Error parsing line: " << line << std::endl;
        }
    }

    inputFile.close();

    // Data in the 'updates' vector; print the data
    for (const auto& update : updates) {
        std::cout << "Timestamp: " << update.timestamp << ", "
                  << "Side: " << update.side << ", "
                  << "Action: " << update.action << ", "
                  << "ID: " << update.id << ", "
                  << "Price: " << update.price << ", "
                  << "Quantity: " << update.quantity << std::endl;
    }

    return 0;
}
