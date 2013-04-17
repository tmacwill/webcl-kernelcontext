$(function() {
    // generate input vectors with 30 random values
    var n = 3;
    var vector = new Uint32Array(n);
    var result = new Uint32Array(n);
    for (var i = 0; i < n; i++)
        vector[i] = Math.floor(Math.random() * 1000);

    // connect to gpu
    var context = new KernelContext;
    var utils = new KernelUtils(context);

    // compute sum in serial
    var sum = 0;
    for (var i = 0; i < vector.length; i++)
        sum += vector[i];
    console.log('Sum Host: ', sum);

    // compute sum reduction
    console.log('Sum Device (reduce): ', utils.reduce(vector, 'a + b'));

    var sumKernel = utils.reductionKernel(Uint32Array, 'a + b');
    var vector_d = context.toGPU(vector);
    console.log('Sum Device (reductionKernel): ' + sumKernel(vector_d, n));

    // compute min reduction
    console.log('Max Host: ', Math.max.apply(null, vector));
    console.log('Max Device: ', utils.reduce(vector, '(a > b) ? a : b'));
});
