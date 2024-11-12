#pragma once

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_FEATURES 30

typedef struct WBCD_t WBCD;

//  Wisconsin Diagnostic Breast Cancer struct
struct WBCD_t {
    int id;
    char diagnosis;
    float features[NUM_FEATURES];
};

/*
    Parses one line into one element

    @param line CSV line of text
    @param values element
*/
void parse_wbcd_line(char* line, WBCD* values) 
{    
    
    char* token = strtok(line, ",");    
    int feature_idx = 0;
    while(token != NULL) {
        if(feature_idx == 0)
        {
            values->id = atoi(token);
        }else if (feature_idx == 1)
        {
            values->diagnosis = token[0];
        }else{
            values->features[feature_idx-2] = atof(token);
        }
        
        token = strtok(NULL, ",");                
        feature_idx = feature_idx + 1;
    }
}

/*
    Calculatest minkowski distance between two elements

    @param a first element
    @param b second element
*/
double minkowski_distance(const WBCD* a, const WBCD* b, int p)
{
     double accumulation = 0;

    for(int feature_idx = 0; feature_idx < NUM_FEATURES; ++feature_idx)
    {
        double feature_difference = fabs(a->features[feature_idx] - b->features[feature_idx]);
        accumulation =  accumulation + pow(feature_difference, p);
    }

    double distance = pow(accumulation, (double)1/p);

    return distance;
}

double ecludian_distance(const WBCD* a, const WBCD* b)
{
    return minkowski_distance(a, b, 2);
}

double manhattan_distance(const WBCD* a, const WBCD* b)
{
    return minkowski_distance(a, b, 1);
}
