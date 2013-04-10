$(function() {
    // generate input vectors with 30 random values
    var n = 30;
    var vector = new Uint32Array(n);
    var result = new Uint32Array(n);
    for (var i = 0; i < n; i++)
        vector[i] = Math.floor(Math.random() * 100);

    // connect to gpu
    var context = new KernelContext;
    var utils = new KernelUtils(context);

    // compute min reduction
    console.log('Min Host: ', Math.min.apply(null, vector));
    console.log('Min Device: ', utils.reduce(vector, '(a < b) ? a : b'));

    // compute sum in serial
    var sum = 0;
    for (var i = 0; i < vector.length; i++)
        sum += vector[i];
    console.log('Sum Host: ', sum);

    // compute sum reduction
    console.log('Sum Device: ', utils.reduce(vector, 'a + b'));
});
