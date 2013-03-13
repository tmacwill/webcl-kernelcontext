function loadKernel(id){
    var kernelElement = document.getElementById(id);
    var kernelSource = kernelElement.text;
    if (kernelElement.src != "") {
        var mHttpReq = new XMLHttpRequest();
        mHttpReq.open('GET', kernelElement.src, false);
        mHttpReq.send(null);
        kernelSource = mHttpReq.responseText;
    }

    return kernelSource;
}

$(function() {
    // write output
    var output = document.getElementById('output');
    output.innerHTML = '';

    try {
        // make sure webcl is supported
        if (window.WebCL == undefined) {
            alert("Unfortunately your system does not support WebCL. " +
                  "Make sure that you have both the OpenCL driver " +
                  "and the WebCL browser extension installed.");
            return false;
        }

        // generate input vectors with 30 random values
        var n = 30;
        var vector1 = new Uint32Array(n);
        var vector2 = new Uint32Array(n);
        for (var i = 0; i < n; i++) {
            vector1[i] = Math.floor(Math.random() * 100);
            vector2[i] = Math.floor(Math.random() * 100);
        }

        // connect to gpu
        var platforms = WebCL.getPlatformIDs();
        var ctx = WebCL.createContextFromType([WebCL.CL_CONTEXT_PLATFORM,
                                               platforms[0]],
                                               WebCL.CL_DEVICE_TYPE_DEFAULT);

        // connect to gpu
        var size = n * 4;
        var buffer1 = ctx.createBuffer(WebCL.CL_MEM_READ_ONLY, size);
        var buffer2 = ctx.createBuffer(WebCL.CL_MEM_READ_ONLY, size);
        var bufferResult = ctx.createBuffer(WebCL.CL_MEM_WRITE_ONLY, size);

        // get kernel source
        var kernelSrc = loadKernel("clVectorAdd");
        var program = ctx.createProgramWithSource(kernelSrc);
        var devices = ctx.getContextInfo(WebCL.CL_CONTEXT_DEVICES);

        // compile kernel
        try {
            program.buildProgram([devices[0]], "");
        }
        catch(e) {
            alert ("Failed to build WebCL program. Error " +
                    program.getProgramBuildInfo(devices[0], WebCL.CL_PROGRAM_BUILD_STATUS) + ": " +
                    program.getProgramBuildInfo (devices[0], WebCL.CL_PROGRAM_BUILD_LOG));
        }

        // send arguments to kernel
        var kernel = program.createKernel('clVectorAdd');
        kernel.setKernelArg(0, buffer1);
        kernel.setKernelArg(1, buffer2);
        kernel.setKernelArg(2, bufferResult);
        kernel.setKernelArg(3, n, WebCL.types.INT);

        // send data to gpu
        var queue = ctx.createCommandQueue(devices[0], 0);
        queue.enqueueWriteBuffer(buffer1, false, 0, size, vector1, []);
        queue.enqueueWriteBuffer(buffer2, false, 0, size, vector2, []);

        // init ND-range
        var localWS = [8];
        var globalWS = [Math.ceil(n / localWS) * localWS];
        output.innerHTML += "<br>Global work item size: " + globalWS;
        output.innerHTML += "<br>Local work item size: " + localWS;

        // execute kernel
        queue.enqueueNDRangeKernel(kernel, globalWS.length, [], globalWS, localWS, []);

        // get result from gpu
        var outBuffer = new Uint32Array(n);
        queue.enqueueReadBuffer(bufferResult, false, 0, size, outBuffer, []);
        queue.finish();

        // display result
        output.innerHTML += "<br>Vector1 = ";
        for (var i = 0; i < n; i++)
            output.innerHTML += vector1[i] + ", ";

        output.innerHTML += "<br>Vector2 = ";
        for (var i = 0; i < n; i++)
            output.innerHTML += vector2[i] + ", ";

        output.innerHTML += "<br>Result = ";
        for (var i = 0; i < n; i++)
            output.innerHTML += outBuffer[i] + ", ";
    }
    catch(e) {
        alert('Error: ' + e.message);
    }
});
