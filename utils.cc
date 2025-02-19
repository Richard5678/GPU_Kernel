int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

float calculateGFLOPS(int m, int n, int s, float elapsed_ms)
{
    // Total floating point operations: 2 * (m * n * s)
    float flops = 2.0f * m * n * s;

    // Convert elapsed time from milliseconds to seconds
    float elapsed_secs = elapsed_ms / 1000.0f;

    // Compute GFLOPS (Giga FLOPS) = (flops / elapsed_secs) / 1e9
    float gflops = (flops / elapsed_secs) / 1e9f;

    return gflops;
}