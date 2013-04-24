$(function() {
    // generate input vectors with 30 random values
    var n = 100;
    var vector = new Uint32Array(n);
    var result = new Uint32Array(n);
    for (var i = 0; i < n; i++)
        vector[i] = Math.floor(Math.random() * 1000);

    // connect to gpu
    var context = new KernelContext;
    var utils = new KernelUtils(context);

    // add 1 to each element in serial
    var serial = new Uint32Array(n);
    for (var i = 0; i < vector.length; i++)
        serial[i] = vector[i] + 1;
    console.log('Map Host: ', serial);

    // add 1 to each element in parallel
    var mapKernel = utils.mapKernel(Uint32Array, 'x', 'x[i] + 1');
    var vector_d = context.toGPU(vector);
    var result_d = context.toGPU(result);
    mapKernel(result_d, n, vector_d);
    context.fromGPU(result_d, result);
    console.log('Map Device (mapKernel): ', result);

    // mapping utility function
    console.log('Map Device (map): ', utils.map('x', 'x[i] + 1', vector));
});
