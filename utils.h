#include <string>
#include <vector>

using namespace std;

struct Card {
    string name;
    vector<double> features;
};

std::vector<std::string> parseCSVRow(const std::string &line);

vector<Card> readCSV(const string &filename);

void writeCSVWithCardData(
    const string &filename,
    const vector<Card> &data,
    const vector<int> &labels,
    const vector<string> &header
);
