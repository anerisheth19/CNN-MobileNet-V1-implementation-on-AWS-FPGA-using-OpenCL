#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>

#define HEIGHT 224
#define WIDTH 224
#define K 3
#define K_P 1
#define FILTER_SIZE_L1 32
#define FILTER_SIZE_L2 32
#define FILTER_SIZE_L3 64
#define FILTER_SIZE_L4 64
#define FILTER_SIZE_L5 128
#define FILTER_SIZE_L6 128
#define FILTER_SIZE_L9 256
#define FILTER_SIZE_L13 512
#define FILTER_SIZE_L25 1024
#define FILTER_SIZE_L29 1000


unsigned char image[HEIGHT * WIDTH * K]; //image with 3 input channels
int decode_image(unsigned char frame[HEIGHT * WIDTH * K], char filename[]);
void readSquezeNetKernel(int *m, int read_size) 
{

	FILE *fp;	
	char buff[255];
	double n;
	fp = fopen("weights_c.txt", "r");
	//int sizeInt = K * K * K * 32 *sizeof(int);
	int i=0;
	for(i = 1; i < read_size + 1; i++)
	{	
		fscanf(fp, "%s", buff);
		n = atof(buff);
		m[i-1] = n;
	}
	fclose(fp);
}
//Function to read image files in C
int decode_image(unsigned char frame[HEIGHT * WIDTH * K],char filename[])
{
	FILE *pFile;
	pFile = fopen(filename, "r"); //read mode
	fseek(pFile, 0, SEEK_SET);
	fread(frame, sizeof(unsigned char), HEIGHT * WIDTH * K, pFile);
	fclose(pFile);
	return 0;
}
//Function to load OpenCL kernel - taken from code given by T.A. Arnab 
long LoadOpenCLKernel(char const* path, char **buf)
{
	FILE  *fp;
	size_t fsz;
	long   off_end;
	int    rc;

	/* Open the file */
	fp = fopen(path, "r");
	if( NULL == fp ) {
		return -1L;
	}

	/* Seek to the end of the file */
	rc = fseek(fp, 0L, SEEK_END);
	if( 0 != rc ) {
		return -1L;
	}

	/* Byte offset to the end of the file (size) */
	if( 0 > (off_end = ftell(fp)) ) {
		return -1L;
	}
	fsz = (size_t)off_end;

	/* Allocate a buffer to hold the whole file */
	*buf = (char *) malloc( fsz+1);
	if( NULL == *buf ) {
		return -1L;
	}

	/* Rewind file pointer to start of file */
	rewind(fp);

	/* Slurp file into buffer */
	if( fsz != fread(*buf, 1, fsz, fp) ) {
		free(*buf);
		return -1L;
	}

	/* Close the file */
	if( EOF == fclose(fp) ) {
		free(*buf);
		return -1L;
	}


	/* Make sure the buffer is NUL-terminated, just in case */
	(*buf)[fsz] = '\0';

	/* Return the file size */
	return (long)fsz;
}
//This is the main function
int main(int argc, char** argv) {

	//define memory for inputs and kernel
	int8_t* filter = (int8_t*) malloc(FILTER_SIZE_L25*FILTER_SIZE_L25*K*K*K*sizeof(int8_t));
	unsigned char* image_r = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //R channel
	unsigned char* image_g = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //G channel
	unsigned char* image_b = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //B channel
	int imagecount  = 0;
	int i,j,k;
	
	int stride = 2;
	int op_size = 32;
	int err;
	cl_device_id device_id;             // compute device id 
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel;                   // compute kernel

	cl_mem d_image_r; //R channel
	cl_mem d_image_g; //G channel
	cl_mem d_image_b; //B channel

	cl_mem d_filter; //filter
	cl_mem d_output; //output image
	cl_event myevent; //profiling event
	cl_ulong start; //time start
	cl_ulong end; //time stop
	cl_float kernelExecTimeNs;

	unsigned char* output_image = (unsigned char*) malloc(FILTER_SIZE_L1 * (HEIGHT/2) * (WIDTH/2) * sizeof(unsigned char));
	
	printf("Initializing OpenCL device...\n"); 

	cl_uint dev_cnt = 0;
	clGetPlatformIDs(0, 0, &dev_cnt);
	
	cl_platform_id platform_ids[100];
	clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	
	// Connect to a compute device
	int gpu = 1;
	err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}
  
	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Create a command commands
	commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	// Create the compute program from the source file
	char *KernelSource;
	long lFileSize;

	lFileSize = LoadOpenCLKernel("kernel.cl", &KernelSource);
	if( lFileSize < 0L ) {
		perror("File read failed");
		return 1;
	}

	program = clCreateProgramWithSource(context, 1, (const char **) &KernelSource, NULL, &err);
	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "convolute", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	decode_image(image,"Cat_Image0.ppm"); //call the read function
    
	//separate R,G and B pixels
	int count = 0;

	for(i = 0;i<HEIGHT * WIDTH * K;i+=3)
	{
		image_r[count] = image[i];
		count++;
	}
	count = 0;

	for(j = 1;j<HEIGHT * WIDTH * K;j+=3)
	{
		image_g[count] = image[j]; 
		count++;
	}
	count = 0;
    
	for(k = 2;k<HEIGHT * WIDTH * K;k+=3)
	{
		image_b[count] = image[k];
		count++; 
	}
    
	//Get filter values
	readSquezeNetKernel(filter, (FILTER_SIZE_L1*K*K*K));

	//Create buffer for device
	d_image_r = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HEIGHT*WIDTH*sizeof(unsigned char), image_r, &err);
	d_image_g = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HEIGHT*WIDTH*sizeof(unsigned char), image_g, &err);
	d_image_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HEIGHT*WIDTH*sizeof(unsigned char), image_b, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L1*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L1*K*K*K*sizeof(int8_t), filter, &err);

	if (!d_image_r || !d_image_g || !d_image_b || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_r, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_r, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands, d_image_g, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_g, 0, NULL, NULL);   
	err = clEnqueueWriteBuffer(commands, d_image_b, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_b, 0, NULL, NULL);   
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L1*K*K*K*sizeof(int8_t), filter, 0, NULL, NULL);   
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	size_t localWorkSize[2], globalWorkSize[2];
	int rows = HEIGHT;
	int cols = WIDTH;
	int filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_r);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_image_g);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_image_b);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&filtersize);
    err |= clSetKernelArg(kernel, 8, sizeof(int), (void *)&stride);
    err |= clSetKernelArg(kernel, 9, sizeof(int), (void *)&op_size);

	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 224;
	globalWorkSize[1] = 224;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L1*(HEIGHT/2)*(WIDTH/2)*sizeof(unsigned char), output_image, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}
     
	//Get kernel execution time
	printf("Kernel Execution time for Layer 1: %f\n",kernelExecTimeNs/1000000000);
	
	/*for (i = 0; i < 20; i++){
                printf("Layer 1 op: %d\t",output_image[i]);
        }
	printf("\n");*/

	//Layer 2 Depth-Wise Convolution

	cl_mem d_image_l2;	//Layer 2 - Input Data
	kernelExecTimeNs = 0;
	op_size = 32;
	stride = 1;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L2*K*K));

	//Create buffer for device
	d_image_l2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L2*sizeof(unsigned char), output_image, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L2*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L2*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l2 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l2, CL_TRUE, 0, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L2*sizeof(unsigned char), output_image, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L2*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/2;
	cols = WIDTH/2;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l2);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 112;
	globalWorkSize[1] = 112;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L1*(HEIGHT/2)*(WIDTH/2)*sizeof(unsigned char), output_image, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	printf("Kernel Execution time for Layer 2: %f\n",kernelExecTimeNs/1000000000);

	/*for (i = 0; i < 20; i++){
                printf("Layer 2 op: %d\t",output_image[i]);
        }
	printf("\n");*/

	//Layer 3 Point-Wise Convolution

	
	unsigned char* output_image_l3 = (unsigned char*) malloc(FILTER_SIZE_L3 * (HEIGHT/2) * (WIDTH/2) * sizeof(unsigned char));

	cl_mem d_image_l3;	//Layer 3 - Input Data
	kernelExecTimeNs = 0;
	op_size = 64;

	readSquezeNetKernel(filter, (K_P*K_P*FILTER_SIZE_L2*FILTER_SIZE_L3));

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	//Create buffer for device
	d_image_l3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L2*sizeof(unsigned char), output_image, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L3*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K_P*K_P*FILTER_SIZE_L2*FILTER_SIZE_L3*sizeof(int8_t), filter, &err);

	if (!d_image_l3 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l3, CL_TRUE, 0, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L2*sizeof(unsigned char), output_image, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, K_P*K_P*FILTER_SIZE_L2*FILTER_SIZE_L3*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/2;
	cols = WIDTH/2;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l3);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
	
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 112;
	globalWorkSize[1] = 112;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L1*(HEIGHT/2)*(WIDTH/2)*sizeof(unsigned char), output_image_l3, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 3: %f\n",kernelExecTimeNs/1000000000);
	
	/*for (i = 0; i < 20; i++){
                printf("Layer 3 op: %d\t",output_image_l3[i]);
        }
	printf("\n");*/

	//Layer 4 Depthwise stride 2
	stride = 2;
	op_size = 64;
	
	unsigned char* output_image_l4 = (unsigned char*) malloc(FILTER_SIZE_L4 * (HEIGHT/4) * (WIDTH/4) * sizeof(unsigned char));

	cl_mem d_image_l4;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L4*K*K));

	//Create buffer for device
	d_image_l4 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L4*sizeof(unsigned char), output_image_l3, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L4*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L4*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l4 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l4, CL_TRUE, 0, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L4*sizeof(unsigned char), output_image_l3, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L4*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/2;
	cols = WIDTH/2;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l4);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 112;
	globalWorkSize[1] = 112;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L4*(HEIGHT/4)*(WIDTH/4)*sizeof(unsigned char), output_image_l4, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 4: %f\n",kernelExecTimeNs/1000000000);
	
	/*for (i = 0; i < 20; i++){
                printf("Layer 4 op: %d\t",output_image_l4[i]);
        }
	printf("\n");*/

	//Layer 5 Pointwise
	
	op_size = 128;
	unsigned char* output_image_l5 = (unsigned char*) malloc(FILTER_SIZE_L5 * (HEIGHT/4) * (WIDTH/4) * sizeof(unsigned char));

	cl_mem d_image_l5;	//Layer 5 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L5*FILTER_SIZE_L4*K_P*K_P));

	//Create buffer for device
	d_image_l5 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L4*sizeof(unsigned char), output_image_l4, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L5*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L5*FILTER_SIZE_L4*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l4 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l5, CL_TRUE, 0, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L4*sizeof(unsigned char), output_image_l4, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L4*FILTER_SIZE_L5*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/4;
	cols = WIDTH/4;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l5);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 56;
	globalWorkSize[1] = 56;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L5*(HEIGHT/4)*(WIDTH/4)*sizeof(unsigned char), output_image_l5, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get execution time
	printf("Kernel Execution time for Layer 5: %f\n",kernelExecTimeNs/1000000000);

	/*for (i = 0; i < 20; i++){
                printf("Layer 6 op: %d\t",output_image_l5[i]);
        }
	printf("\n");*/

	//Layer 6 Depthwise stride 1
	
	op_size = 128;
	stride = 1;
	unsigned char* output_image_l6 = (unsigned char*) malloc(FILTER_SIZE_L6 * (HEIGHT/4) * (WIDTH/4) * sizeof(unsigned char));

	cl_mem d_image_l6;	//Layer 6 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L6*K*K));

	//Create buffer for device
	d_image_l6 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L6*sizeof(unsigned char), output_image_l5, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L6*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L6*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l6 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l6, CL_TRUE, 0, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L6*sizeof(unsigned char), output_image_l5, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L6*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/4;
	cols = WIDTH/4;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l6);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 56;
	globalWorkSize[1] = 56;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L6*(HEIGHT/4)*(WIDTH/4)*sizeof(unsigned char), output_image_l6, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 6: %f\n",kernelExecTimeNs/1000000000);

	/*for (i = 0; i < 20; i++){
                printf("Layer 6 op: %d\t",output_image_l6[i]);
        }
	printf("\n");*/

	//Layer 7 Pointwise
	op_size = 128;
	unsigned char* output_image_l7 = (unsigned char*) malloc(FILTER_SIZE_L6 * (HEIGHT/4) * (WIDTH/4) * sizeof(unsigned char));

	cl_mem d_image_l7;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L6*FILTER_SIZE_L6*K_P*K_P));

	//Create buffer for device
	d_image_l7 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L6*sizeof(unsigned char), output_image_l6, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L6*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L6*FILTER_SIZE_L6*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l7 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l7, CL_TRUE, 0, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L6*sizeof(unsigned char), output_image_l6, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L6*FILTER_SIZE_L6*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/4;
	cols = WIDTH/4;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l7);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 56;
	globalWorkSize[1] = 56;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L6*(HEIGHT/4)*(WIDTH/4)*sizeof(unsigned char), output_image_l7, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 7: %f\n",kernelExecTimeNs/1000000000);

	/*for (i = 0; i < 20; i++){
                printf("Layer 7 op: %d\t",output_image_l7[i]);
        }
	printf("\n");*/
	
	//Layer 8 Depthwise stride 2
	
	unsigned char* output_image_l8 = (unsigned char*) malloc(FILTER_SIZE_L6 * (HEIGHT/4) * (WIDTH/4) * sizeof(unsigned char));

	stride = 2;
	op_size = 128;
	cl_mem d_image_l8;	//Layer 8 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L6*K*K));

	//Create buffer for device
	d_image_l8 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L6*sizeof(unsigned char), output_image_l7, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L6*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L6*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l8 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l8, CL_TRUE, 0, (HEIGHT/4)*(WIDTH/4)*FILTER_SIZE_L6*sizeof(unsigned char), output_image_l7, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L6*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/4;
	cols = WIDTH/4;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l8);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 8;
	localWorkSize[1] = 8;
	globalWorkSize[0] = 56;
	globalWorkSize[1] = 56;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L6*(HEIGHT/8)*(WIDTH/8)*sizeof(unsigned char), output_image_l8, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 8: %f\n",kernelExecTimeNs/1000000000);
	
	/*for (i = 0; i < 20; i++){
                printf("Layer 8 op: %d\t",output_image_l8[i]);
        }
	printf("\n");*/

	//Layer 9 Pointwise
	op_size = 256;
	unsigned char* output_image_l9 = (unsigned char*) malloc(FILTER_SIZE_L9 * (HEIGHT/8) * (WIDTH/8) * sizeof(unsigned char));

	cl_mem d_image_l9;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L6*FILTER_SIZE_L9*K_P*K_P));

	//Create buffer for device
	d_image_l9 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L6*sizeof(unsigned char), output_image_l8, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L9*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L6*FILTER_SIZE_L9*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l9 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l9, CL_TRUE, 0, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L6*sizeof(unsigned char), output_image_l8, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L6*FILTER_SIZE_L9*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/8;
	cols = WIDTH/8;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l9);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 28;
	globalWorkSize[1] = 28;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L9*(HEIGHT/8)*(WIDTH/8)*sizeof(unsigned char), output_image_l9, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 9: %f\n",kernelExecTimeNs/1000000000);

	/*for (i = 0; i < 20; i++){
                printf("Layer 9 op: %d\t",output_image_l9[i]);
        }
	printf("\n");*/

	//Layer 10 Depthwise stride 1
	op_size = 256;
	stride = 1;
	unsigned char* output_image_l10 = (unsigned char*) malloc(FILTER_SIZE_L9 * (HEIGHT/8) * (WIDTH/8) * sizeof(unsigned char));

	cl_mem d_image_l10;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L9*K*K));

	//Create buffer for device
	d_image_l10 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L9*sizeof(unsigned char), output_image_l9, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L9*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L9*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l10 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l10, CL_TRUE, 0, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L9*sizeof(unsigned char), output_image_l9, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L9*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/8;
	cols = WIDTH/8;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l10);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 28;
	globalWorkSize[1] = 28;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L9*(HEIGHT/8)*(WIDTH/8)*sizeof(unsigned char), output_image_l10, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 10: %f\n",kernelExecTimeNs/1000000000);

	/*for (i = 0; i < 20; i++){
                printf("Layer 10 op: %d\t",output_image_l10[i]);
        }
	printf("\n");*/

	//Layer 11 Pointwise
	op_size = 256; 
	unsigned char* output_image_l11 = (unsigned char*) malloc(FILTER_SIZE_L9 * (HEIGHT/8) * (WIDTH/8) * sizeof(unsigned char));

	cl_mem d_image_l11;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L9*FILTER_SIZE_L9*K_P*K_P));

	//Create buffer for device
	d_image_l11 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L9*sizeof(unsigned char), output_image_l10, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L9*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L9*FILTER_SIZE_L9*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l11 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l11, CL_TRUE, 0, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L9*sizeof(unsigned char), output_image_l10, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L9*FILTER_SIZE_L9*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/8;
	cols = WIDTH/8;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l11);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 28;
	globalWorkSize[1] = 28;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L9*(HEIGHT/8)*(WIDTH/8)*sizeof(unsigned char), output_image_l11, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 11: %f\n",kernelExecTimeNs/1000000000);
	
	/*for (i = 0; i < 20; i++){
                printf("Layer 11 op: %d\t",output_image_l11[i]);
        }
        printf("\n");*/
	
	//Layer 12 Depthwise stride 2
	stride = 2;
	op_size = 256;
	
	unsigned char* output_image_l12 = (unsigned char*) malloc(FILTER_SIZE_L9 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l12;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L9*K*K));

	//Create buffer for device
	d_image_l12 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L9*sizeof(unsigned char), output_image_l11, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L9*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L9*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l12 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l12, CL_TRUE, 0, (HEIGHT/8)*(WIDTH/8)*FILTER_SIZE_L9*sizeof(unsigned char), output_image_l11, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L9*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/8;
	cols = WIDTH/8;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l12);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 28;
	globalWorkSize[1] = 28;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L9*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l12, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 12: %f\n",kernelExecTimeNs/1000000000);
	
	/*for (i = 0; i < 20; i++){
                printf("Layer 10 op: %d\t",output_image_l12[i]);
        }
        printf("\n");*/

	//Layer 13 Pointwise
	op_size = 512;
	
	unsigned char* output_image_l13 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l13;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L9*FILTER_SIZE_L13*K_P*K_P));

	//Create buffer for device
	d_image_l13 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L9*sizeof(unsigned char), output_image_l11, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L9*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l13 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l13, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L9*sizeof(unsigned char), output_image_l11, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L9*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/16;
	cols = WIDTH/16;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l13);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 14;
	globalWorkSize[1] = 14;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l13, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 13: %f\n",kernelExecTimeNs/1000000000);
	
	/*for (i = 0; i < 20; i++){
                printf("Layer 13 op: %d\t",output_image_l13[i]);
        }
        printf("\n");*/

	//Layer 14 Depthwise stride 1
	op_size = 512;
	stride = 1;
	
	unsigned char* output_image_l14 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l14;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L13*K*K));

	//Create buffer for device
	d_image_l14 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l13, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l14 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l14, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l13, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/16;
	cols = WIDTH/16;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l14);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 14;
	globalWorkSize[1] = 14;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l14, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 14: %f\n",kernelExecTimeNs/1000000000);
	
	/*for (i = 0; i < 20; i++){
                printf("Layer 14 op: %d\t",output_image_l14[i]);
        }
        printf("\n");*/

	//Layer 15 Pointwise
	op_size = 512;
	
	unsigned char* output_image_l15 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l15;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P));

	//Create buffer for device
	d_image_l15 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l14, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l15 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l15, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l14, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/16;
	cols = WIDTH/16;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l15);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 14;
	globalWorkSize[1] = 14;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l15, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 15: %f\n",kernelExecTimeNs/1000000000);

	/*for (i = 0; i < 20; i++){
                printf("Layer 15 op: %d\t",output_image_l15[i]);
        }
        printf("\n");*/

	//Layer 16 Depthwise stride 1
	stride = 1;
	op_size = 512;
	
	unsigned char* output_image_l16 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l16;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L13*K*K));

	//Create buffer for device
	d_image_l16 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l15, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l16 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l16, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l15, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/16;
	cols = WIDTH/16;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l16);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 14;
	globalWorkSize[1] = 14;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l16, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 16: %f\n",kernelExecTimeNs/1000000000);

	/*for (i = 0; i < 20; i++){
                printf("Layer 16 op: %d\t",output_image_l16[i]);
        }
        printf("\n");*/

	//Layer 17 Pointwise
	op_size = 512;
	
	unsigned char* output_image_l17 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l17;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P));

	//Create buffer for device
	d_image_l17 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l16, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l17 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l17, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l16, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/16;
	cols = WIDTH/16;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l17);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 14;
	globalWorkSize[1] = 14;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l17, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 17: %f\n",kernelExecTimeNs/1000000000);

	//Layer 18 Depthwise stride 1
	stride = 1;
	op_size = 512;
	
	unsigned char* output_image_l18 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l18;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L13*K*K));

	//Create buffer for device
	d_image_l18 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l17, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l18 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l18, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l17, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/16;
	cols = WIDTH/16;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l18);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 14;
	globalWorkSize[1] = 14;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l18, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 18: %f\n",kernelExecTimeNs/1000000000);

	//Layer 19 Pointwise
	op_size = 512;
	
	unsigned char* output_image_l19 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l19;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P));

	//Create buffer for device
	d_image_l19 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l18, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l19 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l19, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l18, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/16;
	cols = WIDTH/16;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l15);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 14;
	globalWorkSize[1] = 14;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l19, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 19: %f\n",kernelExecTimeNs/1000000000);

	//Layer 20 Depthwise stride 1
	stride = 1;
	op_size = 512;
	
	unsigned char* output_image_l20 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l20;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L13*K*K));

	//Create buffer for device
	d_image_l20 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l19, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l20 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l20, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l19, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/16;
	cols = WIDTH/16;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l14);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 14;
	globalWorkSize[1] = 14;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l20, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 20: %f\n",kernelExecTimeNs/1000000000);

	//Layer 21 Pointwise
	op_size = 512;
	
	unsigned char* output_image_l21 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l21;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P));

	//Create buffer for device
	d_image_l21 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l20, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l21 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l21, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l20, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/16;
	cols = WIDTH/16;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l21);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 14;
	globalWorkSize[1] = 14;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l21, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 21: %f\n",kernelExecTimeNs/1000000000);

	//Layer 22 Depthwise stride 1
	stride = 1;
	op_size = 512;
	
	unsigned char* output_image_l22 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l22;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L13*K*K));

	//Create buffer for device
	d_image_l22 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l21, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l22 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l22, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l21, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/16;
	cols = WIDTH/16;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l22);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 14;
	globalWorkSize[1] = 14;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l22, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 22: %f\n",kernelExecTimeNs/1000000000);

	//Layer 23 Pointwise
	op_size = 512;
	
	unsigned char* output_image_l23 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/16) * (WIDTH/16) * sizeof(unsigned char));

	cl_mem d_image_l23;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P));

	//Create buffer for device
	d_image_l23 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l22, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l23 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l23, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l22, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L13*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/16;
	cols = WIDTH/16;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l23);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 14;
	globalWorkSize[1] = 14;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/16)*(WIDTH/16)*sizeof(unsigned char), output_image_l23, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 23: %f\n",kernelExecTimeNs/1000000000);

	//Layer 24 Depthwise stride 2
	stride = 2;
	op_size = 512;
	
	unsigned char* output_image_l24 = (unsigned char*) malloc(FILTER_SIZE_L13 * (HEIGHT/32) * (WIDTH/32) * sizeof(unsigned char));

	cl_mem d_image_l24;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L13*K*K));

	//Create buffer for device
	d_image_l24 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l23, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/32)*(WIDTH/32)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l24 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l24, CL_TRUE, 0, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l23, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L13*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/32;
	cols = WIDTH/32;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l24);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 7;
	globalWorkSize[1] = 7;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L13*(HEIGHT/32)*(WIDTH/32)*sizeof(unsigned char), output_image_l24, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 24: %f\n",kernelExecTimeNs/1000000000);

	//Layer 25 Pointwise
	op_size = 1024;
	
	unsigned char* output_image_l25 = (unsigned char*) malloc(FILTER_SIZE_L25 * (HEIGHT/32) * (WIDTH/32) * sizeof(unsigned char));

	cl_mem d_image_l25;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L25*FILTER_SIZE_L13*K_P*K_P));

	//Create buffer for device
	d_image_l25 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/32)*(WIDTH/32)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l24, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/32)*(WIDTH/32)*FILTER_SIZE_L25*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L25*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l25 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l25, CL_TRUE, 0, (HEIGHT/32)*(WIDTH/32)*FILTER_SIZE_L13*sizeof(unsigned char), output_image_l24, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L25*FILTER_SIZE_L13*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/32;
	cols = WIDTH/32;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l25);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 7;
	globalWorkSize[1] = 7;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L25*(HEIGHT/32)*(WIDTH/32)*sizeof(unsigned char), output_image_l25, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 25: %f\n",kernelExecTimeNs/1000000000);

	//Layer 26 Depthwise stride 1
	stride = 2;
	op_size = 1024;
	
	unsigned char* output_image_l26 = (unsigned char*) malloc(FILTER_SIZE_L25 * (HEIGHT/32) * (WIDTH/32) * sizeof(unsigned char));

	cl_mem d_image_l26;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "depthwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L25*K*K));

	//Create buffer for device
	d_image_l26 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/32)*(WIDTH/32)*FILTER_SIZE_L25*sizeof(unsigned char), output_image_l25, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/16)*(WIDTH/16)*FILTER_SIZE_L13*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L25*K*K*sizeof(int8_t), filter, &err);	

	if (!d_image_l26 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l26, CL_TRUE, 0, (HEIGHT/32)*(WIDTH/32)*FILTER_SIZE_L25*sizeof(unsigned char), output_image_l25, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L25*K*K*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/32;
	cols = WIDTH/32;
	filtersize = K;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l26);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&stride);
	err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 7;
	globalWorkSize[1] = 7;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L25*(HEIGHT/32)*(WIDTH/32)*sizeof(unsigned char), output_image_l26, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 26: %f\n",kernelExecTimeNs/1000000000);

	//Layer 27 Pointwise
	op_size = 1024;
	
	unsigned char* output_image_l27 = (unsigned char*) malloc(FILTER_SIZE_L25 * (HEIGHT/32) * (WIDTH/32) * sizeof(unsigned char));

	cl_mem d_image_l27;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L25*FILTER_SIZE_L25*K_P*K_P));

	//Create buffer for device
	d_image_l27 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/32)*(WIDTH/32)*FILTER_SIZE_L25*sizeof(unsigned char), output_image_l26, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/32)*(WIDTH/32)*FILTER_SIZE_L25*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L25*FILTER_SIZE_L25*K_P*K_P*sizeof(int8_t), filter, &err);	

	if (!d_image_l27 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l27, CL_TRUE, 0, (HEIGHT/32)*(WIDTH/32)*FILTER_SIZE_L25*sizeof(unsigned char), output_image_l26, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L25*FILTER_SIZE_L25*K_P*K_P*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/32;
	cols = WIDTH/32;
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l27);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 7;
	localWorkSize[1] = 7;
	globalWorkSize[0] = 7;
	globalWorkSize[1] = 7;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L25*(HEIGHT/32)*(WIDTH/32)*sizeof(unsigned char), output_image_l27, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 27: %f\n",kernelExecTimeNs/1000000000);

	//Layer 28 Average Pooling Stride 7

	op_size = 1024;
	unsigned char* output_image_l28 = (unsigned char*) malloc(FILTER_SIZE_L25 * (HEIGHT/224) * (WIDTH/224) * sizeof(unsigned char));

	cl_mem d_image_l28;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pool", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}


	//Create buffer for device
	d_image_l28 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/32)*(WIDTH/32)*FILTER_SIZE_L25*sizeof(unsigned char), output_image_l27, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/224)*(WIDTH/224)*FILTER_SIZE_L25*sizeof(unsigned char), NULL, &err);

	if (!d_image_l28 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l28, CL_TRUE, 0, (HEIGHT/32)*(WIDTH/32)*FILTER_SIZE_L25*sizeof(unsigned char), output_image_l27, 0, NULL, NULL);
	 
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/32;
	cols = WIDTH/32;
	filtersize = 7;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l28);
	err |= clSetKernelArg(kernel, 2, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&op_size);
	
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 1;
	localWorkSize[1] = 1;
	globalWorkSize[0] = 7;
	globalWorkSize[1] = 7;
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L25*(HEIGHT/224)*(WIDTH/224)*sizeof(unsigned char), output_image_l28, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Layer 28: %f\n",kernelExecTimeNs/1000000000);

	//Layer 29 Fully Connected
	op_size = 1000;
	unsigned char* output_image_l29 = (unsigned char*) malloc(FILTER_SIZE_L29 * (HEIGHT/224) * (WIDTH/224) * sizeof(unsigned char));

	cl_mem d_image_l29;	//Layer 4 - Input Data
	kernelExecTimeNs = 0;

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, "pointwise", &err);
	if (!kernel || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	readSquezeNetKernel(filter, (FILTER_SIZE_L25*FILTER_SIZE_L29));

	//Create buffer for device
	d_image_l29 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (HEIGHT/224)*(WIDTH/224)*FILTER_SIZE_L25*sizeof(unsigned char), output_image_l28, &err);
	d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (HEIGHT/224)*(WIDTH/224)*FILTER_SIZE_L29*sizeof(unsigned char), NULL, &err);
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L25*FILTER_SIZE_L29*sizeof(int8_t), filter, &err);	

	if (!d_image_l27 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l29, CL_TRUE, 0, (HEIGHT/224)*(WIDTH/224)*FILTER_SIZE_L25*sizeof(unsigned char), output_image_l28, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L25*FILTER_SIZE_L29*sizeof(int8_t), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/224;	
	cols = WIDTH/224;	
	filtersize = K_P;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l29);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&op_size);
    
	if (err != CL_SUCCESS)
	{ 
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	localWorkSize[0] = 1;	//Need to change
	localWorkSize[1] = 1;
	globalWorkSize[0] = 1;
	globalWorkSize[1] = 1;

	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
    
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to execute kernel! %d\n", err);
		exit(1);
	}
   
	clWaitForEvents(1,&myevent);	 
	clFinish(commands);   
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	kernelExecTimeNs += end - start;
	err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, FILTER_SIZE_L29*(HEIGHT/224)*(WIDTH/224)*sizeof(unsigned char), output_image_l29, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	//Get kernel execution time
	printf("Kernel Execution time for Fully Connected Layer: %f\n",kernelExecTimeNs/1000000000);
	
	/*for (i = 0; i < 20; i++){
		printf("Layer 29 op: %d\t",output_image_l29[i]);
	}*/
	
	//Layer 30 - Softmax
	
	double* output_softmax = (double*) malloc(FILTER_SIZE_L29 * sizeof(double));
	double sum = 0.0;
	int loop, location;
	for (loop = 0; loop < FILTER_SIZE_L29; loop++){
		output_softmax[loop] = exp(output_image_l29[loop]);
		sum += exp(output_image_l29[loop]);
	}
	for (loop = 0; loop < FILTER_SIZE_L29; loop++){
		output_softmax[loop] = output_softmax[loop]/sum;
		//printf("SOFTMAX OP: %f\t", output_softmax[loop]);
	}
	
	double maximum = output_softmax[0];
 	
	for (loop = 1; loop < FILTER_SIZE_L29; loop++) {
		if (output_softmax[loop] > maximum) {
			maximum  = output_softmax[loop];
       			location = loop + 1;
		}
	}
	
	printf("Highest Probability of the element is present at location %d and it's value is %f.\n", location, maximum);
	
	//Shutdown and cleanup
	free(image_r);
	free(image_g);
	free(image_b);
	free(filter);
	free(output_image);
	free(output_image_l3);free(output_image_l4);free(output_image_l5);
 	free(output_image_l5);free(output_image_l6);free(output_image_l7);
	free(output_image_l8);free(output_image_l9);free(output_image_l10);
	free(output_image_l11);free(output_image_l12);free(output_image_l13);
	free(output_image_l14);free(output_image_l15);free(output_image_l16);
	free(output_image_l17);free(output_image_l18);free(output_image_l19);
	free(output_image_l20);free(output_image_l21);free(output_image_l22);
	free(output_image_l23);free(output_image_l24);free(output_image_l25);
	free(output_image_l26);free(output_image_l27);free(output_image_l28);free(output_image_l29);free(output_softmax);
	clReleaseMemObject(d_image_r);
	clReleaseMemObject(d_image_g);
	clReleaseMemObject(d_image_b);
	clReleaseMemObject(d_image_l2);
	clReleaseMemObject(d_image_l3);
	clReleaseMemObject(d_image_l4);
	clReleaseMemObject(d_image_l5);	
	clReleaseMemObject(d_image_l6);
	clReleaseMemObject(d_image_l7);
	clReleaseMemObject(d_image_l8);
	clReleaseMemObject(d_image_l9);
	clReleaseMemObject(d_image_l10);
	clReleaseMemObject(d_image_l11);
	clReleaseMemObject(d_image_l12);clReleaseMemObject(d_image_l13);clReleaseMemObject(d_image_l14);clReleaseMemObject(d_image_l15);
	clReleaseMemObject(d_image_l16);clReleaseMemObject(d_image_l17);clReleaseMemObject(d_image_l18);clReleaseMemObject(d_image_l19);clReleaseMemObject(d_image_l20);
	clReleaseMemObject(d_image_l21);clReleaseMemObject(d_image_l22);clReleaseMemObject(d_image_l23);clReleaseMemObject(d_image_l24);clReleaseMemObject(d_image_l25);
	clReleaseMemObject(d_image_l26);clReleaseMemObject(d_image_l27);clReleaseMemObject(d_image_l28);clReleaseMemObject(d_image_l29);
	clReleaseMemObject(d_output);
	clReleaseMemObject(d_filter);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
	return 0;
}
