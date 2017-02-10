/*
Copyright 2016 Narrative Nights Inc. All Rights Reserved.
Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_jni.h"

#include <math.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>

#include <jni.h>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/stat_summarizer.h"
#include "jni_utils2.h"

static std::unique_ptr<tensorflow::Session> session;

static bool g_compute_graph_initialized = false;
using namespace tensorflow;

JNIEXPORT jint JNICALL
TENSORFLOW_METHOD(initTS)(JNIEnv* env,
						jobject thiz,
						jobject java_asset_manager,
						jstring model) {
	
// 	if (g_compute_graph_initialized) {
// 		LOG(INFO) << "Compute graph already loaded. skipping.";
// 		return 0;
// 	}
	
	const char* const model_cstr = env->GetStringUTFChars(model, NULL);
	
	LOG(INFO) << "Loading Tensorflow.";
	LOG(INFO) << "Making new SessionOptions.";
	
	tensorflow::SessionOptions options;
	tensorflow::ConfigProto& config = options.config;
	LOG(INFO) << "Got config, " << config.device_count_size() << " devices";
	session.reset(tensorflow::NewSession(options));
	LOG(INFO) << "Session created.";
	
	tensorflow::GraphDef graph_def;
	
	LOG(INFO) << "Graph created.";
	
	AAssetManager* const asset_manager = AAssetManager_fromJava(env, java_asset_manager);
	
	LOG(INFO) << "Acquired AssetManager.";

	LOG(INFO) << "Reading file to proto: " << model_cstr;
	
	ReadFileToProto2(asset_manager, model_cstr, &graph_def);

	LOG(INFO) << "Creating session.";

	tensorflow::Status s = session->Create(graph_def);
	
	if (!s.ok()) {
		LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
		return -1;
	}

	// Clear the proto to save memory space.
	graph_def.Clear();
	
	LOG(INFO) << "Tensorflow graph loaded from: " << model_cstr;

	g_compute_graph_initialized = true;

	return 0;
}

static int* process2(const int* pixels, const int height, const int width, const int outHeight, const int outWidth, int* result) {
	LOG(INFO) << "mapping input";

	// Create input tensor
	Tensor input_tensor( tensorflow::DT_FLOAT, tensorflow::TensorShape( {height,width,3} ) );
	auto input_tensor_mapped = input_tensor.tensor<float, 3>();
    int i = 0;
	for(int y= 0; y<height; ++y) {
        for(int x= 0; x<width; ++x) {
            int pixel = pixels[i++];
            float red = ((pixel >> 16) & 0xff)/255.0f;
            float green = ((pixel >> 8) & 0xff)/255.0f;
            float blue = ((pixel) & 0xff)/255.0f;
            input_tensor_mapped(y,x,0) = red;
            input_tensor_mapped(y,x,1) = green;
            input_tensor_mapped(y,x,2) = blue;
        }
	}
	//shape input
	Tensor shape_tensor( tensorflow::DT_INT32, tensorflow::TensorShape( {3} ) );
	auto shape_tensor_mapped = shape_tensor.tensor<int, 1>();
	shape_tensor_mapped(0) = height;
	shape_tensor_mapped(1) = width;
	shape_tensor_mapped(2) = 3;

	LOG(INFO) << "Start computing.";

	std::vector<std::pair<std::string, tensorflow::Tensor> > input_tensors({{"input", input_tensor},{"input_shape", shape_tensor}});

	// Actually run the image through the model.
	std::vector<Tensor> output_tensors;
	std::vector<std::string> output_names({"output"});

	Status run_status = session->Run( input_tensors, output_names,
									  {},
									  &output_tensors );
	
	LOG(INFO) << "End computing.";
	
	if (!run_status.ok()) {
		LOG(ERROR) << "Error during inference: " << run_status;
		return NULL;
	}
	
	Tensor& output_tensor = output_tensors[0];
// 	tensorflow::TTypes<float>::Flat output_flat = output_tensor.flat<float>();
// 	LOG(INFO) << "flattened";

	auto output_mapped = output_tensor.tensor<float, 3>();
	
	int h = height;
	int w = width;
	if(outHeight > 0){
		h = outHeight;
	}
	if(outWidth > 0){
		w = outWidth;
	}
	
    i=0;
	for(int y = 0; y < h; ++y) {
        for(int x = 0; x < w; ++x) {
        	int red = round(output_mapped(y,x,0)*255);
        	int green = round(output_mapped(y,x,1)*255);
        	int blue = round(output_mapped(y,x,2)*255);
        	if(red>255){
        		red=255;
			} else if(red<0){
				red=0;
			}
			if(green>255){
        		green=255;
			} else if(green<0){
				green=0;
			}
			if(blue>255){
        		blue=255;
			} else if(blue<0){
				blue=0;
			}
        	int pixel = (0xFF << 24) | (red << 16) | (green << 8) | blue;
            result[i++] = pixel;
        }
	}

	LOG(INFO) << "copied to the array";
	
	return result;
}

JNIEXPORT jintArray JNICALL
TENSORFLOW_METHOD(execTSGraph)(JNIEnv* env, jobject thiz, jintArray raw_pixels, jint height, jint width) {
	jboolean iCopied = JNI_FALSE;
	jint* pixels = env->GetIntArrayElements(raw_pixels, &iCopied);

	int result[height*width] = {};
	process2( reinterpret_cast<int*>(pixels), (int) height, (int) width, 0, 0 , result);
	env->ReleaseIntArrayElements(raw_pixels, pixels, JNI_ABORT);

	LOG(INFO) << "returned from process and released input memory";

    jintArray jresult;
    jresult = env->NewIntArray(height*width);
    if (jresult == NULL) {
         return NULL; /* out of memory error thrown */
    }
    env->SetIntArrayRegion(jresult, 0, height*width, result);

	LOG(INFO) << "all finished, prepare to return";

	return jresult;
}


JNIEXPORT jintArray JNICALL
TENSORFLOW_METHOD(execTSGraph2)(JNIEnv* env, jobject thiz, jintArray raw_pixels, jint height, jint width, jint outHeight, jint outWidth) {
	jboolean iCopied = JNI_FALSE;
	jint* pixels = env->GetIntArrayElements(raw_pixels, &iCopied);

	int result[outHeight * outWidth] = {};
	process2( reinterpret_cast<int*>(pixels), (int) height, (int) width, (int) outHeight, (int) outWidth , result);
	env->ReleaseIntArrayElements(raw_pixels, pixels, JNI_ABORT);

	LOG(INFO) << "returned from process and released input memory";

    jintArray jresult;
    jresult = env->NewIntArray(outHeight * outWidth);
    if (jresult == NULL) {
         return NULL; /* out of memory error thrown */
    }
    env->SetIntArrayRegion(jresult, 0, outHeight * outWidth, result);

	LOG(INFO) << "all finished, prepare to return";

	return jresult;
}
