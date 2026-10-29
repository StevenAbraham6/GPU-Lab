// No fast-relaxed-math. FMA is explicit and counts as two floating-point ops.
__kernel void arithmetic_fp(__global float4 *out, uint iterations) {
    uint i = get_global_id(0);
    float4 v = (float4)(0.01f*(i%17), 0.02f*(i%13), 0.03f*(i%11), 0.04f*(i%7));
    for (uint j=0; j<iterations; ++j)
        v = fma(v, (float4)(1.000001f), (float4)(0.000001f));
    out[i] = v;
}
__kernel void arithmetic_int(__global uint4 *out, uint iterations) {
    uint i = get_global_id(0);
    uint4 v = (uint4)(i, i+1, i+2, i+3);
    for (uint j=0; j<iterations; ++j)
        v = v * (uint4)(1664525u) + (uint4)(1013904223u);
    out[i] = v;
}
__kernel void read_memory(__global const float *a, __global float *out, uint n) {
    uint i = get_global_id(0), stride=get_global_size(0);
    float v=0;
    for (uint j=i; j<n; j+=stride) v += a[j];
    out[i]=v;
}
__kernel void write_memory(__global float *a, uint n) {
    uint i=get_global_id(0), stride=get_global_size(0);
    for (uint j=i; j<n; j+=stride) a[j]=(float)(j%1024);
}
__kernel void copy_memory(__global const float *a, __global float *b, uint n, uint step) {
    // step is odd and n is a power of two: access is a permutation, no omitted words.
    uint i=get_global_id(0), stride=get_global_size(0);
    for (uint j=i; j<n; j+=stride) {
        uint k=(j*step)&(n-1); b[k]=a[k];
    }
}
__kernel void chase(__global const uint *links, __global uint *out, uint steps) {
    uint pos=0;
    for (uint i=0; i<steps; ++i) pos=links[pos];
    out[0]=pos;
}
__kernel void tiny(__global uint *out) { out[0] = 42u; }
uint pattern_at(uint i, uint pattern) {
    if(pattern==0) return 0;
    if(pattern==1) return 0xffffffffu;
    if(pattern==2) return (i&1)?0xaaaaaaaau:0x55555555u;
    uint v=i ^ (pattern*0x9e3779b9u); v ^= v>>16; v*=0x85ebca6bu; v^=v>>13; return v;
}
__kernel void fill_pattern(__global uint *a, uint n, uint pattern) {
    for(uint i=get_global_id(0); i<n; i+=get_global_size(0)) a[i]=pattern_at(i,pattern);
}
__kernel void verify_pattern(__global const uint *a, __global uint *counts, uint n, uint pattern) {
    uint errors=0;
    for(uint i=get_global_id(0); i<n; i+=get_global_size(0)) errors += (a[i]!=pattern_at(i,pattern));
    counts[get_global_id(0)]=errors;
}
