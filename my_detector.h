/*
how to compile
g++ -g -I /usr/include/python3.6m -o b b.cpp -lpython3.6m
*/
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numpy/arrayobject.h>
using namespace std;
struct timeval st_time, end_time;
class Interface{
public:
	Interface (char* path){
		int i, j, k;
		Py_Initialize();
		PyRun_SimpleString("import sys\nsys.path.append('/home/hanwenchen/controllers/my_controller3')\n");
		PyRun_SimpleString("from keras.models import *\nfrom keras.layers import *\nfrom keras.optimizers import *\nfrom model import *\nfrom keras import backend as K\nimport numpy as np\nimport copy\nimport sys\nfrom keras.applications.vgg16 import preprocess_input\nimport time\nimport cv2");
		
		pymodule = PyImport_ImportModule("predict2");
		pyclass = PyObject_GetAttrString(pymodule, "predictor");
		//printf("ok\n");
		FILE *out = fopen("path.temp", "w");
		fprintf(out, "%s\n", path);
		fclose(out);
		//printf("ok\n");
		mypredictor = PyObject_CallObject(pyclass, NULL);
	}

	double query_TF(char* path){
	//takes about 0.13s
		int i, j, k;
		FILE *out = fopen("path.temp", "w");
		fprintf(out, "%s\n", path);
		fclose(out);

		PyObject* pyargs = PyTuple_New(1);
		PyTuple_SetItem(pyargs, 0, Py_BuildValue("O", mypredictor));
		PyObject* pyquery_TF = PyObject_GetAttrString(pyclass, "query_TF");
		PyObject_CallObject(pyquery_TF, pyargs);
		FILE *in = fopen("rslt.temp", "r");
		double res;
		fscanf(in, "%lf", &res);
		fclose(in);
		return res;
	}

	int query_pos(char* path){
		int i, j, k;
		FILE *out = fopen("path.temp", "w");
		fprintf(out, "%s\n", path);
		fclose(out);
		//printf("ok");

		PyObject* pyargs = PyTuple_New(1);
		PyTuple_SetItem(pyargs, 0, Py_BuildValue("O", mypredictor));
		PyObject* pyquery_pos = PyObject_GetAttrString(pyclass, "query_pos");
		PyObject_CallObject(pyquery_pos, pyargs);

		FILE *in = fopen("rslt.temp", "r");
		int res;
		fscanf(in, "%d", &res);
		fclose(in);
		return res;
	}
private:
	PyObject *pymodule, *mypredictor, *pyclass;
};

///////
//testing begins
///////
/*
int templ[2100][2100][3], image[2100][2100][3];

int main(){
	int i, j, n, m;
	//FILE *in = fopen("testtemplate.dat", "r");
	//fscanf(in, "%d%d", &n, &m);
	//for (i = 0; i < n; i++)
	//	for (j = 0; j < m; j++)
	//		fscanf(in, "%d%d%d", &templ[i][j][0], &templ[i][j][1], &templ[i][j][2]);
	//fclose(in);
	char path[50] = "template5.png";
	gettimeofday(&st_time, NULL);
	Interface a(path);
	//in = fopen("image.dat", "r");
	//fscanf(in, "%d%d", &n, &m);
	//for (i = 0; i < n; i++)
	//	for (j = 0; j < m; j++)
	//		fscanf(in, "%d%d%d", &image[i][j][0], &image[i][j][1], &image[i][j][2]);
	//fclose(in);
	gettimeofday(&end_time, NULL);
	//printf("%lld\n", (end_time.tv_sec-st_time.tv_sec) * 1000000 + end_time.tv_usec- st_time.tv_usec);
	char path2[50] = "03.png";
	//printf("%s\n", path);
	int x = a.query_TF(path2);
	gettimeofday(&end_time, NULL);
	//printf("%lld\n", (end_time.tv_sec-st_time.tv_sec) * 1000000 + (end_time.tv_usec - st_time.tv_usec));
	printf("%d\n", x);
	return 0;
}*/
