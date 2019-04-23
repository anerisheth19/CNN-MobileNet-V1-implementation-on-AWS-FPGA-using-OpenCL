### OpenCL "simple_demo" Makefile
### Assumes preferred CUDA environment is loaded

CC = gcc
CFLAGS = -l OpenCL 
TARGET = out
default: $(TARGET)

$(TARGET):
	$(CC) $(CFLAGS) -o $(TARGET) MobileNet.c -lm

clean:
	rm $(TARGET)
