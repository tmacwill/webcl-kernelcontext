var KernelUtils = (function() {
    var reduceSource = "__kernel void reduce(__global $TYPE* buffer, __const int length, __global $TYPE* result) { \
        local $TYPE scratch[$N] = { $BASE }; \
        int global_index = get_global_id(0); \
        int local_index = get_local_id(0); \
        if (global_index < length) \
            scratch[local_index] = buffer[global_index]; \
        else \
            scratch[local_index] = $BASE; \
        barrier(CLK_LOCAL_MEM_FENCE); \
        for (int offset = 1; offset < get_local_size(0); offset <<= 1) { \
            int mask = (offset << 1) - 1; \
            if ((local_index & mask) == 0) { \
                $TYPE a = scratch[local_index]; \
                $TYPE b = scratch[local_index + offset]; \
                scratch[local_index] = $OP; \
            } \
            barrier(CLK_LOCAL_MEM_FENCE); \
        } \
        if (local_index == 0) \
            result[get_group_id(0)] = scratch[0]; \
    }";

    /**
     * Constructor
     *
     */
    var KernelUtils = function(context) {
        // save kernel context
        this.context = context;
    };

    /**
     * Return a CL type for the given data
     * @param data {Array} Array to get data type for
     *
     */
    var type = function(data) {
        if (data instanceof Float64Array || data instanceof Float32Array)
            return 'float';
        if (data instanceof Uint32Array || data instanceof Uint16Array || Uint8Array)
            return 'unsigned int';
        if (data instanceof Int32Array || data instanceof Int16Array || data instanceof Int8Array)
            return 'int';
    };

    /**
     * Perform a reduction
     * @param vector {Array} Array to reduce
     * @param op {String} Operation to perform on array
     * @param base {Number} Optional base case for the reduction
     * @param local {Number} Optional local memory size
     *
     */
    KernelUtils.prototype.reduce = function(vector, op, base, local) {
        // default base case is 0
        if (base === undefined)
            base = 0;

        // default value for local work size
        if (local === undefined)
            local = 32;
        var global = Math.ceil(vector.length / local) * local;

        // make sure input array is a power of 2
        var n = Math.pow(2, Math.ceil(Math.log(vector.length) / Math.log(2)));
        if (vector.length != n) {
            // create a new array with length 2^m padded with the base case
            var padded = new vector.constructor(n);
            for (var i = 0; i < n; i++)
                padded[i] = (i < vector.length) ? vector[i] : base;
            vector = padded;
        }

        // send input and output to gpu
        var result = new Uint32Array(1);
        var vector_d = this.context.toGPU(vector);
        var result_d = this.context.toGPU(result);

        // compile kernel, replacing placeholders with input values
        var source = reduceSource.replace(/\$N/g, vector.length).replace(/\$TYPE/g, type(vector))
            .replace(/\$OP/g, op).replace(/\$BASE/g, base);
        var kernel = this.context.compile(source, 'reduce');

        // execute kernel
        kernel({
            local: local,
            global: global
        }, vector_d, new Uint32(vector.length), result_d);

        // get result from GPU
        this.context.fromGPU(result_d, result);
        return result[0];
    };

    return KernelUtils;
})();
