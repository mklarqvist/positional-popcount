#include <stdio.h>
#include <stdlib.h>

#include "fast_flagstats.h"

int main() {
    uint16_t datain[] = {14,64,12,1923,12621,1203,4129,12314,12,1124,12314};
    int N = sizeof(datain) / sizeof(uint16_t);

    uint32_t counters[16] = {0};
    pospopcnt_u16(datain,N,counters);

    printf("counts:");
    for(int i = 0; i < 16; ++i) {
        printf(" %d", counters[i]);
    }
    printf("\n");
}