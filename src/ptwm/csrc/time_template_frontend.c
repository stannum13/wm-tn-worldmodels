#include <math.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const double *detector_weights;
    const double *pair_weights;
    const int32_t *pairs;
    const double *bias;
    const double *risks;
    size_t detector_count;
    size_t pair_count;
    int32_t rule;
    double joint[20];
    double posterior[4];
    double logits[4];
} TimeProgram;

void time_program_reset(TimeProgram *p) {
    for (int k = 0; k < 20; ++k) p->joint[k] = 0.05;
    for (int m = 0; m < 4; ++m) p->posterior[m] = 0.25;
}

int time_program_choose(TimeProgram *p, const uint8_t *detectors) {
    static const double rates[5] = {0.005, 0.02, 0.1, 0.5, 0.75};
    size_t count = 0;
    for (int m = 0; m < 4; ++m) p->logits[m] = p->bias[m];
    for (size_t i = 0; i < p->detector_count; ++i) {
        if (detectors[i]) {
            ++count;
            for (int m = 0; m < 4; ++m)
                p->logits[m] += p->detector_weights[4 * i + m];
        }
    }
    for (size_t i = 0; i < p->pair_count; ++i) {
        if (detectors[p->pairs[2 * i]] && detectors[p->pairs[2 * i + 1]]) {
            for (int m = 0; m < 4; ++m)
                p->logits[m] += p->pair_weights[4 * i + m];
        }
    }
    double mode_mass[4] = {0, 0, 0, 0};
    double maximum = p->logits[0];
    for (int m = 0; m < 4; ++m) {
        maximum = fmax(maximum, p->logits[m]);
        for (int k = 0; k < 5; ++k) mode_mass[m] += p->joint[4 * k + m];
    }
    double emission[4];
    for (int m = 0; m < 4; ++m)
        emission[m] = exp(fmax(-700.0, p->logits[m] - maximum));
    double normalizer = 0;
    for (int k = 0; k < 5; ++k) {
        double mixed[4], total = 0;
        for (int m = 0; m < 4; ++m) {
            mixed[m] = 0.998 * p->joint[4 * k + m] + 0.002 * mode_mass[m] / 5.0;
            total += mixed[m];
        }
        for (int m = 0; m < 4; ++m) {
            p->joint[4 * k + m] = ((1.0 - rates[k] - rates[k] / 3.0) * mixed[m]
                                   + rates[k] * total / 3.0) * emission[m];
            normalizer += p->joint[4 * k + m];
        }
    }
    for (int m = 0; m < 4; ++m) {
        p->posterior[m] = 0;
        for (int k = 0; k < 5; ++k) {
            p->joint[4 * k + m] /= normalizer;
            p->posterior[m] += p->joint[4 * k + m];
        }
    }
    int chosen = 0;
    if (p->rule == 0) {
        for (int m = 1; m < 4; ++m)
            if (p->posterior[m] > p->posterior[chosen]) chosen = m;
        return chosen;
    }
    const int bin = p->rule == 1 ? 0 : (count == 0 ? 0 : (count < 4 ? 1 : (count < 8 ? 2 : 3)));
    double best = INFINITY;
    for (int a = 0; a < 4; ++a) {
        double risk = 0;
        for (int m = 0; m < 4; ++m)
            risk += p->posterior[m] * p->risks[(4 * m + a) * 4 + bin];
        if (risk < best) {
            best = risk;
            chosen = a;
        }
    }
    return chosen;
}
