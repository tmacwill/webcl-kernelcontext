/**
 * KernelContext Number, which has a value and a type
 *
 */
var KernelContextNumber = (function() {
    /**
     * Constructor
     *
     * @param {Number} n Value of number
     * @param {String} type Type of number
     *
     */
    var KernelContextNumber = function(n, type) {
        this.n = n;
        this.type = type;
    };

    /**
     * Conform to native Number interface
     *
     */
    KernelContextNumber.prototype.valueOf = function() {
        return this.n;
    };

    return KernelContextNumber;
})();

/**
 * Scalar type definitions
 *
 */
var Char = function(n) { return new KernelContextNumber(n, 'Char'); };
var Float = function(n) { return new KernelContextNumber(n, 'Float'); };
var Int32 = function(n) { return new KernelContextNumber(n, 'Int32'); };
var Int16 = function(n) { return new KernelContextNumber(n, 'Int16'); };
var Int8 = function(n) { return new KernelContextNumber(n, 'Int8'); };
var Uint32 = function(n) { return new KernelContextNumber(n, 'Uint32'); };
var Uint16 = function(n) { return new KernelContextNumber(n, 'Uint16'); };
var Uint8 = function(n) { return new KernelContextNumber(n, 'Uint8'); };

var KernelContext = (function() {
    /**
     * Object representing an executable WebCL kernel
     * To run, simply call this like a function, and it will be executed on the GPU
     *
     * @param {Object} kernelContext Reference to context object that generated this kernel
     * @param {Object} kernel Compiled kernel object to be executed
     *
     */
    var KernelContextKernel = function(kernelContext, kernel) {
        // store references to context object and compiled kernel so we can use them later
        this.kernelContext = kernelContext;
        this.kernel = kernel;

        // return function that when executed (with arguments) will run kernel on GPU
        var self = this;
        return function() {
            // block and grid parameters
            var params = arguments[0];

            // make sure that params are arrays
            if (!(params.global instanceof Array))
                params.global = [params.global];
            if (!(params.local instanceof Array))
                params.local = [params.local];

            // set each argument (to this inner function) as a kernel argument
            for (var i = 1; i < arguments.length; i++) {
                var type = webClType(arguments[i]);
                if (type)
                    self.kernel.setKernelArg(i - 1, arguments[i].valueOf(), type);
                else
                    self.kernel.setKernelArg(i - 1, arguments[i]);
            }

            // execute kernel
            self.kernelContext.queue.enqueueNDRangeKernel(self.kernel, params.global.length, [], params.global, params.local, []);
        };
    };

    /**
     * Constructor
     * Connect to GPU if WebCL is supported
     *
     */
    var KernelContext = function() {
        // make sure webcl is supported
        if (window.WebCL == undefined) {
            throw "Unfortunately your system does not support WebCL. " +
                  "Make sure that you have both the OpenCL driver " +
                  "and the WebCL browser extension installed.";
        }

        // connect to gpu
        this.platforms = WebCL.getPlatformIDs();
        this.context = WebCL.createContextFromType([WebCL.CL_CONTEXT_PLATFORM,
                                                    this.platforms[0]],
                                                    WebCL.CL_DEVICE_TYPE_DEFAULT);

        // initialize command queue
        this.devices = this.context.getContextInfo(WebCL.CL_CONTEXT_DEVICES);
        this.queue = this.context.createCommandQueue(this.devices[0], 0);

        // variables currently on the gpu
        this.onGPU = [];
    };

    /**
     * Get the number of bytes in a data object
     *
     * @param {Object} Data Object to get size of
     * @return {Number} Size, in bytes, of data
     *
     */
    var bytes = function(data) {
        if (data instanceof Float64Array)
            return data.length * 8;
        if (data instanceof Uint32Array || data instanceof Int32Array || data instanceof Float32Array)
            return data.length * 4;
        else if (data instanceof Uint16Array || data instanceof Int16Array)
            return data.length * 2;
        else if (data instanceof Uint8Array || data instanceof Int8Array)
            return data.length;
    };

    /**
     * Get the WebCL type from a built-in scalar type
     *
     * @param {Object} n Scalar type for which to get WebCL constant
     * @return {Number} WebCL constant
     *
     */
    var webClType = function(n) {
        // map of custom wrapper class names to webcl constants
        var scalars = {
            'Char': WebCL.types.CHAR,
            'Float': WebCL.types.FLOAT,
            'Int32': WebCL.types.INT,
            'Int16': WebCL.types.SHORT,
            'Int8': WebCL.types.BYTE,
            'Uint32': WebCL.types.UINT,
            'Uint16': WebCL.types.USHORT,
            'Uint8': WebCL.types.UBYTE
        };

        return scalars[n.type];
    };

    /**
     * Compile a program from a source code string
     *
     * @param {String} source Source code to compile
     * @param {String} f Name of function inside source code to compile
     *
     */
    KernelContext.prototype.compile = function(source, f) {
        // get kernel source
        var program = this.context.createProgramWithSource(source);

        // compile kernel
        try {
            program.buildProgram([this.devices[0]], "");
        }
        catch(e) {
            console.log("Failed to build WebCL program. Error " +
                        program.getProgramBuildInfo(this.devices[0], WebCL.CL_PROGRAM_BUILD_STATUS) + ": " +
                        program.getProgramBuildInfo(this.devices[0], WebCL.CL_PROGRAM_BUILD_LOG));
        }

        // create a new kernel object from compiled source
        return new KernelContextKernel(this, program.createKernel(f));
    };

    /**
     * Read data from the GPU
     *
     * @param {Object} handle Handle to read from
     * @param {Object} result Optional pre-allocated object to read data into
     * @return {Object} Data from GPU
     *
     */
    KernelContext.prototype.fromGPU = function(handle, result) {
        // if result is not pre-allocated, then look up type and allocate space
        if (result === undefined)
            for (var i = 0; i < this.onGPU.length; i++)
                if (this.onGPU[i].handle == handle)
                    result = new this.onGPU[i].data.constructor(this.onGPU[i].data.length);

        // enqueue the buffer for reading and execute queue so read occurs
        var size = bytes(result);
        this.queue.enqueueReadBuffer(handle, false, 0, size, result, []);
        this.queue.finish();

        return result;
    };

    /**
     * Send data to the GPU
     *
     * @param {Object} data Data to send to GPU
     * @param {Number} size Length of data
     * @return {Object} Handle to data on GPU
     *
     */
    KernelContext.prototype.toGPU = function(data) {
        // create and enqueue a new buffer for writing
        var size = bytes(data);
        var buffer = this.context.createBuffer(WebCL.CL_MEM_READ_WRITE, size);
        this.queue.enqueueWriteBuffer(buffer, false, 0, size, data, []);

        // remember this variable has been sent to the GPU
        this.onGPU.push({
            data: data,
            handle: buffer
        });

        return buffer;
    };

    return KernelContext;
})();
