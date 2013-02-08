// cuda kernel
var source = "__kernel void clVectorAdd(__global unsigned int* a, __global unsigned int* b, __global unsigned int* result, unsigned int width) { \
     unsigned int x = get_global_id(0); \
     if (x >= width) \
       return; \
    result[x] = a[x] + b[x]; \
}";

$(function() {
    // write output
    var output = document.getElementById('output');
    output.innerHTML = '';

    // generate input vectors with 30 random values
    var n = 30;
    var vector1 = new Uint32Array(n);
    var vector2 = new Uint32Array(n);
    var result = new Uint32Array(n);
    for (var i = 0; i < n; i++) {
        vector1[i] = Math.floor(Math.random() * 100);
        vector2[i] = Math.floor(Math.random() * 100);
    }

    // connect to gpu
    var tmcl = new TMCL();

    // compile kernel from source
    //var vector_kernel = tmcl.compile(source, 'clVectorAdd');
    tmcl.compile(source, 'clVectorAdd');

    // send data to gpu
    var vector1Handle = tmcl.toGPU(vector1);
    var vector2Handle = tmcl.toGPU(vector2)
    var resultHandle = tmcl.toGPU(result);

    // run kernel
    var local = 8;
    var global = Math.ceil(n / local) * local;
    tmcl.kernels.clVectorAdd({
    //vector_kernel({
        local: local,
        global: global
    }, vector1Handle, vector2Handle, resultHandle, new Uint32(n));

    // read result from gpu
    //var r = tmcl.fromGPU(resultHandle);
    tmcl.fromGPU(resultHandle, result);

    // display result
    output.innerHTML += "<br>Vector1 = ";
    for (var i = 0; i < n; i++)
        output.innerHTML += vector1[i] + ", ";

    output.innerHTML += "<br>Vector2 = ";
    for (var i = 0; i < n; i++)
        output.innerHTML += vector2[i] + ", ";

    output.innerHTML += "<br>Result = ";
    for (var i = 0; i < n; i++)
        output.innerHTML += result[i] + ", ";
});
