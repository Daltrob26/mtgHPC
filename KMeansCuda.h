#include <vector>
#include "utils.h"

std::vector<int> kMeansCUDA(
    const std::vector<Card>& data,                        
    int k, 
    int max_iters,
    const int blocks=126
);