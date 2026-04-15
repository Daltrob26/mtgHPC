#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "utils.h"

using namespace std;


std::vector<std::string> parseCSVRow(const std::string &line) {
    std::vector<std::string> result;
    std::string cur;
    bool inQuotes = false;

    for (char c : line) {
        // handle cards with , in the name
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            result.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    result.push_back(cur);
    return result;
}

std::vector<Card> readCSV(const std::string &filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<Card> data;

    bool firstLine = true;

    while (std::getline(file, line)) {
        // skip the header line
        if (firstLine) {
            firstLine = false;
            continue;
        }

        auto cols = parseCSVRow(line);

        Card card;
        card.name = cols[0];

        for (size_t i = 1; i < cols.size(); ++i) {
            card.features.push_back(std::stod(cols[i]));
        }

        data.push_back(card);
    }

    return data;
}

// write out the data, final col is the cluster the card belongs to
void writeCSVWithCardData(
    const std::string &filename,
    const std::vector<Card> &data,
    const std::vector<int> &labels,
    const std::vector<std::string> &header
) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    // Write header
    for (size_t i = 0; i < header.size(); ++i) {
        out << '"' << header[i] << '"' << ",";
    }
    out << "\"cluster\"\n";

    // Write rows
    for (size_t i = 0; i < data.size(); ++i) {
        out << '"' << data[i].name << '"';
        for (double f : data[i].features) {
            out << "," << f;
        }
        out << "," << labels[i] << "\n";
    }

    out.close();
}