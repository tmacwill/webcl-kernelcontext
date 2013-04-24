var KernelUtils = (function() {
    // $N = block_dim (local_size)
    // length = size of buffer <= #threads
    // Launched with #threads rounded to multiple of block_dim (pad with $BASE)
    var reduceSource = "__kernel void reduce(__global $TYPE* buffer, __const int length, __global $TYPE* result) { \
        local $TYPE scratch[$N]; \
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
     * Return a CL type for the given Javascript type
     * @param t {Function} Javascript type to get CL type for
     *
     */
    var type = function(t) {
        if (t == Float64Array)
            return 'double'
        if (t == Float32Array)
            return 'float';
        if (t == Uint32Array || t == Uint16Array || Uint8Array)
            return 'unsigned int';
        if (t == Int32Array || t == Int16Array || t == Int8Array)
            return 'int';
    };

    /**
     * Construct a reduction kernel
     * @param t {Function} Type of array to reduce
     * @param op {String} Operation to perform on array
     * @param base {Number} Optional base case for the reduction
     * @param local {Number} Optional local memory size
     * @return Function that can be called with two arguments: handle to device, length of vector
     *
     */
    KernelUtils.prototype.reductionKernel = function(t, op, base, local) {
        // default base case is 0
        if (base === undefined)
            base = 0;

        // default value for local work size
        if (local === undefined)
            local = 32;

        // compile kernel, replacing placeholders with input values
        var source = reduceSource.replace(/\$N/g, local).replace(/\$TYPE/g, type(t))
            .replace(/\$OP/g, op).replace(/\$BASE/g, base);
        var kernel = this.context.compile(source, 'reduce');

        var self = this;
        return function(vector_d, length, result_d) {
            // make sure vector length is a multiple of local size
            var n = length;
            if (n % local != 0)
                n = length + local - (length % local);

            // allocate result and send to GPU
            if (result_d === undefined) {
                var result = new t(Math.ceil(n / local));
                result_d = self.context.toGPU(result);
            }

            do {
                // execute kernel
                kernel({
                    local: local,
                    global: Math.ceil(n / local) * local
                }, vector_d, new Uint32(n), result_d);

                // make sure vector length is a multiple of local size
                n = Math.ceil(n / local);
                if (n % local != 0)
                    n = n + local - (n % local);

                // point next input at current output
                vector_d = result_d;
            }
            while (Math.ceil(n / local) > 1);

            // get final answer from gpu
            var reduced = new t(1);
            self.context.fromGPU(result_d, reduced);
            return reduced[0];
        };
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
        var vector_d = this.context.toGPU(vector);
        var kernel = this.reductionKernel(vector.constructor, op, base, local);
        return kernel(vector_d, vector.length);
    };

    return KernelUtils;
})();
