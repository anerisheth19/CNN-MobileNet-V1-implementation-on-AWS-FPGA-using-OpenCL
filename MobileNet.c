//ECGR 6090 Heterogeneous Computing Homework 2
// Problem 3  - Direct Convolution in OpenCL for SqueezeNet
//Written by Aneri Sheth - 801085402
//Reference taken from Lecture Slides by Dr. Tabkhi and code given by TA Arnab Purkayastha

#include <fcntl.h>
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

unsigned char image[HEIGHT * WIDTH * K]; //image with 3 input channels
int decode_image(unsigned char frame[HEIGHT * WIDTH * K], char filename[]);
//void read_filters(int *filters);
void readSquezeNetKernel(int *m, int read_size) 
{

	FILE *fp;	
	char buff[255];
	double n;
	fp = fopen("sample_weights.txt", "r");
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
	int* filter = (int*) malloc(FILTER_SIZE_L1*K*K*K*sizeof(int));
	unsigned char* image_r = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //R channel
	unsigned char* image_g = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //G channel
	unsigned char* image_b = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //B channel
	//unsigned char* image_input = (unsigned char*) malloc(3 * HEIGHT * WIDTH * sizeof(unsigned char)); //all channels together

	int imagecount  = 0;
	int i,j,k;

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
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L1*K*K*K*sizeof(int), filter, &err);

	if (!d_image_r || !d_image_g || !d_image_b || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_r, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_r, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commands, d_image_g, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_g, 0, NULL, NULL);   
	err = clEnqueueWriteBuffer(commands, d_image_b, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_b, 0, NULL, NULL);   
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L1*K*K*K*sizeof(int), filter, 0, NULL, NULL);   
   
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

	/*int l;
	
	int image_c = 0;
	for(l = 0; l < (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L1; l++){
		printf("%d\n", output_image[l]);
	
	}*/	

	//Layer 2 Depth-Wise Convolution

	cl_mem d_image_l2;	//Layer 2 - Input Data
	kernelExecTimeNs = 0;

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
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, FILTER_SIZE_L2*K*K*sizeof(int), filter, &err);	

	if (!d_image_l2 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l2, CL_TRUE, 0, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L2*sizeof(unsigned char), output_image, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, FILTER_SIZE_L2*K*K*sizeof(int), filter, 0, NULL, NULL);
   
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

	//Layer 3 Point-Wise Convolution

	unsigned char* output_image_l3 = (unsigned char*) malloc(FILTER_SIZE_L3 * (HEIGHT/2) * (WIDTH/2) * sizeof(unsigned char));

	cl_mem d_image_l3;	//Layer 3 - Input Data
	kernelExecTimeNs = 0;

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
	d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K_P*K_P*FILTER_SIZE_L2*FILTER_SIZE_L3*sizeof(int), filter, &err);

	if (!d_image_l3 || !d_filter || !d_output)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
	
	err = clEnqueueWriteBuffer(commands, d_image_l3, CL_TRUE, 0, (HEIGHT/2)*(WIDTH/2)*FILTER_SIZE_L2*sizeof(unsigned char), output_image, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, K_P*K_P*FILTER_SIZE_L2*FILTER_SIZE_L3*sizeof(int), filter, 0, NULL, NULL);
   
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write data to device! %d\n", err);
		exit(1);
	}
 
	rows = HEIGHT/2;
	cols = WIDTH/2;
	filtersize = FILTER_SIZE_L3;
    
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image_l3);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
    
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

	printf("Kernel Execution time for Layer 3: %f\n",kernelExecTimeNs/1000000000);

	//Shutdown and cleanup
	free(image_r);
	free(image_g);
	free(image_b);
	free(filter);
	free(output_image);
	free(output_image_l3);
 
	clReleaseMemObject(d_image_r);
	clReleaseMemObject(d_image_g);
	clReleaseMemObject(d_image_b);
	clReleaseMemObject(d_image_l2);
	clReleaseMemObject(d_image_l3);
	clReleaseMemObject(d_output);
	clReleaseMemObject(d_filter);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	return 0;
}
