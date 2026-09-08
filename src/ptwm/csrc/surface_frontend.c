#include <math.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const double *detector_weights;
    const int32_t *pairs;
    const double *pair_weights;
    size_t detector_count;
    size_t pair_count;
    double bias;
    double probability;
    double switch_probability;
    double threshold;
} SurfaceProgram;

double surface_score(const SurfaceProgram *p, const uint8_t *row) {
    double score = p->bias;
    for (size_t i = 0; i < p->detector_count; i++) {
        score += p->detector_weights[i] * row[i];
    }
    for (size_t i = 0; i < p->pair_count; i++) {
        score += p->pair_weights[i] * (row[p->pairs[2*i]] & row[p->pairs[2*i+1]]);
    }
    return score;
}

int surface_choose(SurfaceProgram *p, const uint8_t *row) {
    double prior = p->switch_probability + (1.0 - 2.0*p->switch_probability)*p->probability;
    double odds = surface_score(p, row) + log(prior) - log1p(-prior);
    if (odds >= 0.0) {
        p->probability = 1.0/(1.0 + exp(-odds));
    } else {
        double e = exp(odds);
        p->probability = e/(1.0 + e);
    }
    return p->probability >= p->threshold;
}

/* Unnormalized Walsh-Hadamard transform. n must be a power of two. */
void surface_fwht(double *values, size_t n) {
    for (size_t width = 1; width < n; width *= 2) {
        for (size_t start = 0; start < n; start += 2*width) {
            for (size_t j = 0; j < width; j++) {
                double a = values[start+j];
                double b = values[start+j+width];
                values[start+j] = a+b;
                values[start+j+width] = a-b;
            }
        }
    }
}
