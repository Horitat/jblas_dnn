package dnn;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import layers.Convolutionlayer;
import layers.Hiddenlayer;
import layers.MaxPoolinglayer;

import org.jblas.FloatMatrix;

import single_layer.Logistic_kaiki;
import util.Common_method;
import util.RandomGenerator;
import Mersenne.Sfmt;

public class CNN_test {

	int[] kernelnum;
	int[][] convkernelsize;
	int[][] poolkernelsize;
	int hidden_N;
	int output_N;

	/*
	 * 本来ならレイヤーのスーパークラスを作り、各レイヤーはそのクラスを継承
	 * スーパークラスの型を宣言し、ほしいレイヤーを格納する
	 */
	Convolutionlayer[] conv;
	MaxPoolinglayer[] pool;
	int[][] convoutsize;
	int[][] pooloutsize;
	int flatsize;
	Hiddenlayer hidden;
	Logistic_kaiki logistic;

	Sfmt mt;
	Scanner scan;

	public CNN_test(int[] imageSize, int channel, int[] nKernels, int minibatch,
			int[][] kernelSizes, int[][] poolSizes, int nHidden, int nOut,
			Sfmt m, String activation) {
		// TODO 自動生成されたコンストラクター・スタブ
		scan = new Scanner(System.in);
		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}
		mt = m;

		kernelnum = nKernels;
		convkernelsize = kernelSizes;
		poolkernelsize = poolSizes;
		hidden_N = nHidden;
		output_N = nOut;

		conv = new Convolutionlayer[kernelnum.length];
		pool = new MaxPoolinglayer[conv.length];
		convoutsize = new int[nKernels.length][imageSize.length];
		pooloutsize = new int[nKernels.length][imageSize.length];

		//畳込み層とプーリング層の初期化
		for(int i=0; i<nKernels.length; i++){
			int[] size;
			int chnl;

			if(i==0){
				size = new int[]{imageSize[0], imageSize[1]};
				chnl = channel;
			}else{
				size = new int[]{pooloutsize[i-1][0], pooloutsize[i-1][1]};
				chnl = nKernels[i-1];
			}

			//出力サイズ
			//outputsize = ((inputsize-kernelsize+2*paddingsize)/stridesize)+1
			convoutsize[i] = new int[]{size[0] - kernelSizes[i][0] + 1, size[1] - kernelSizes[i][1] + 1};
			pooloutsize[i] = new int[]{(((convoutsize[i][0]-poolSizes[i][0]+(2*0)) / 1)+1), (((convoutsize[i][0]-poolSizes[i][0]+(2*0)) / 1)+1)};
			System.out.println("convout:"+ (size[0] - kernelSizes[i][0] + 1)+","+ (size[1] - kernelSizes[i][1] + 1));
//			System.out.println("poolout:"+ convoutsize[i][0] / poolSizes[i][0] +","+ convoutsize[i][1] / poolSizes[i][1]);
			System.out.println("poolout1:"+ (((convoutsize[i][0]-poolSizes[i][0]+(2*0)) / 1)+1) +","+ (((convoutsize[i][0]-poolSizes[i][0]+(2*0)) / 1)+1));

			conv[i] = new Convolutionlayer(size,chnl, nKernels[i], kernelSizes[i],convoutsize[i], 1,minibatch, mt, activation);
			pool[i] = new MaxPoolinglayer(poolkernelsize[i],pooloutsize[i], nKernels[i], 1, minibatch,mt, "MAX", "");
		}

		//入力データを一次元に直すためのサイズ、全結合層への入力に使われる
		flatsize = nKernels[nKernels.length-1] * pooloutsize[pooloutsize.length-1][0]* pooloutsize[pooloutsize.length-1][1];
		System.out.println("flatsize:"+nKernels[nKernels.length-1] +":"+ pooloutsize[pooloutsize.length-1][0] +":"+ pooloutsize[pooloutsize.length-1][1]);

		hidden = new Hiddenlayer(flatsize, hidden_N, null, null,minibatch, mt, activation);
		logistic = new Logistic_kaiki(hidden_N, output_N);
	}


	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		Sfmt mt = new Sfmt(init_key);

		int train_N_each = 50;        // for demo
		int test_N_each = 10;          // for demo
		double pNoise_Training = 0.05;  // for demo
		double pNoise_Test = 0.10;     // for demo

		final int patterns = 3;

		final int train_N = train_N_each * patterns;
		final int test_N = test_N_each * patterns;

		final int[] imageSize = {12,12};
		final int channel = 1;

		int[] nKernels = {10, 20};//カーネルの数
		int[][] kernelSizes = { {3, 3}, {3, 3} };//カーネルサイズ
		int[][] poolSizes = { {3, 3}, {2, 2} };//プーリングのカーネルサイズ

		int nHidden = 20;
		final int nOut = patterns;

		FloatMatrix[][] train_X = new FloatMatrix[train_N][channel];
		FloatMatrix train_T = new FloatMatrix(train_N,nOut);

		FloatMatrix[][] test_X = new FloatMatrix[test_N][channel];
		FloatMatrix test_T = new FloatMatrix(test_N,nOut);


		int epochs = 500;
		float learningRate = 0.1f;

		final int minibatchSize = 25;
		int minibatch_N = train_N / minibatchSize;

		for (int n = 0; n < train_N; n++) {
			for (int c = 0; c < channel; c++) {
				train_X[n][c] = new FloatMatrix(imageSize[0],imageSize[1]);
			}
		}

		for (int n = 0; n < test_N; n++) {
			for (int c = 0; c < channel; c++) {
				test_X[n][c] = new FloatMatrix(imageSize[0],imageSize[1]);
			}
		}
		//
		// Create training data and test data for demo.
		//
		float data=0;
		for (int pattern = 0; pattern < patterns; pattern++) {

			for (int n = 0; n < train_N_each; n++) {

				int n_ = pattern * train_N_each + n;

				for (int c = 0; c < channel; c++) {

					for (int i = 0; i < imageSize[0]; i++) {

						for (int j = 0; j < imageSize[1]; j++) {

							if ((i < (pattern + 1) * (imageSize[0] / patterns)) && (i >= pattern * imageSize[0] / patterns)) {
								train_X[n_][c].put(i,j,(float) (((int) 128. * mt.NextUnif() + 128.) * RandomGenerator.binomial(1, 1 - pNoise_Training, mt) / 256.));
							} else {
								train_X[n_][c].put(i,j,(float) (128. * RandomGenerator.binomial(1, pNoise_Training, mt) / 256.));
							}
							//System.out.println(data);
							data += 1.f;
						}
					}
				}

				for (int i = 0; i < nOut; i++) {
					if (i == pattern) {
						train_T.put(n_,i,1);
					} else {
						train_T.put(n_,i,0);
					}
				}
			}
			data = 0.f;
			for (int n = 0; n < test_N_each; n++) {

				int n_ = pattern * test_N_each + n;

				for (int c = 0; c < channel; c++) {

					for (int i = 0; i < imageSize[0]; i++) {

						for (int j = 0; j < imageSize[1]; j++) {

							if ((i < (pattern + 1) * imageSize[0] / patterns) && (i >= pattern * imageSize[0] / patterns)) {
								test_X[n_][c].put(i,j,(float) (((int) 128. * mt.NextUnif() + 128.) * RandomGenerator.binomial(1, 1 - pNoise_Test, mt) / 256.));
							} else {
								test_X[n_][c].put(i,j,(float) (128. * RandomGenerator.binomial(1, pNoise_Test, mt) / 256.));
							}
							data += 1.f;
						}
					}
				}

				for (int i = 0; i < nOut; i++) {
					if (i == pattern) {
						test_T.put(n_,i,1);
					} else {
						test_T.put(n_,i, 0);
					}
				}
			}
		}

		// create minibatches
		FloatMatrix[][][] train_X_minibatch = new FloatMatrix[minibatch_N][minibatchSize][channel];
		FloatMatrix[] train_T_minibatch = new FloatMatrix[minibatch_N];//[minibatchSize][nOut];
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
		Common_method.list_shuffle(minibatchIndex, mt);

		for (int i = 0; i < minibatch_N; i++) {
			train_T_minibatch[i] = new FloatMatrix(minibatchSize,nOut);
			for(int j=0; j<minibatchSize;j++){
				train_T_minibatch[i].putRow(j,train_T.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
		}

		for (int j = 0; j < minibatchSize; j++) {
			for (int i = 0; i < minibatch_N; i++) {
				for(int c=0; c<channel;c++){
					train_X_minibatch[i][j][c] = new FloatMatrix(imageSize[0],imageSize[1]);
					for(int s=0; s<imageSize[0]; s++){
						train_X_minibatch[i][j][c].putRow(s, train_X[minibatchIndex.get(i * minibatchSize + j)][c].getRow(s));
					}
				}
			}
		}


		//
		// Build Convolutional Neural Networks model
		//
		long start = System.currentTimeMillis();
		// construct CNN
		System.out.println("Building the jblas model...");
		CNN_test classifier = new CNN_test(imageSize, channel, nKernels,minibatchSize, kernelSizes, poolSizes, nHidden, nOut, mt, "ReLU");
		System.out.println("done.");

		//System.exit(-1);
		// train the model
		System.out.println("Training the model...");
		System.out.println();

		for (int epoch = 0; epoch < epochs; epoch++) {

			if ((epoch + 1) % 50 == 0) {
				long end = System.currentTimeMillis();
				System.out.print("\titer = " + (epoch + 1) + " / " + epochs);
				System.out.println("  time = "+(end - start)/1000 + "s");
			}

			for (int batch = 0; batch < minibatch_N; batch++) {
				classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate);
			}
			learningRate *= 0.999;
		}
		System.out.println("done. End");

		// test
		FloatMatrix predicted_T = new FloatMatrix(test_N,nOut);
		for(int i=0; i<predicted_T.rows; i++){
			predicted_T.putRow(i,classifier.predict(test_X[i]));
		}
		//
		// Evaluate the model
		//
		long end = System.currentTimeMillis();
		System.out.println("Total time = "+(end - start)/1000 + "s");

		Common_method.print_result(predicted_T, test_T);
		System.out.println("JBlas DNN, conv and pooling layers have each input data and output data");
	}



	/**
	 *
	 * @param x [minibatchsize][chanel][imgsize][imgsize]
	 * @param label
	 * @param minibatchsize
	 * @param l_rate
	 */
	public void train(FloatMatrix[][] x, FloatMatrix label, int minibatchsize, float l_rate){
		//隠れ層の出力キャッシュ用
		FloatMatrix hiddendata = new FloatMatrix(minibatchsize,hidden_N);

		//batchnormがありかなしかで条件分けし、ミニバッチでループか各層をミニバッチ単位でループするか
		for(int n=0; n<minibatchsize; n++){
		//順伝播x[minibatchSize][channel]
			FloatMatrix[] z = x[n].clone();
			for(int i=0; i<kernelnum.length; i++){
				z = conv[i].forward(z,n);
				z = pool[i].maxpooling(z,n);
			}

			FloatMatrix xx = data_flat(z);
			hiddendata.putRow(n, hidden.forward(xx.dup(), n));
		}

		//出力層の逆伝播
		FloatMatrix dx_flat = new FloatMatrix(minibatchsize,flatsize);
		//出力層の順伝播逆伝播
		FloatMatrix dy = logistic.train(hiddendata, label, minibatchsize, l_rate);
		//隠れ層の順伝播逆伝播
		FloatMatrix dz = hidden.backward(dy, logistic.weight, minibatchsize, l_rate);
		//畳込み、プーリングの逆伝播
		FloatMatrix[][] dx = new FloatMatrix[minibatchsize][kernelnum[kernelnum.length-1]];

		//畳込み層、プーリング層の逆伝播のために、フラット化したデータを戻す
		dx_flat = dz.mmul(hidden.weight);
		for(int n=0; n<minibatchsize; n++){
			dx[n] = data_unflat(dx_flat.getRow(n));
		}

		//畳込み層の逆伝播
		FloatMatrix[][] dc = dx.clone();
		for(int i= kernelnum.length-1; i>-1; i--){
			FloatMatrix[][] poolback = pool[i].backmaxpooing(dc, convoutsize[i], minibatchsize);
			dc = conv[i].backward( poolback, minibatchsize, l_rate);
		}
	}


	private FloatMatrix predict(FloatMatrix[] x) {
		FloatMatrix[] z = x.clone();
		for(int i=0; i<kernelnum.length; i++){
			z = conv[i].forward(z, -1);
			z = pool[i].maxpooling(z, -1);
		}
		return logistic.predict(hidden.forward(data_flat(z)));
	}


	private FloatMatrix data_flat(FloatMatrix[] z) {
		FloatMatrix f = new FloatMatrix(1,flatsize);
		int n=0,p=0;
		for(int k=0; k<kernelnum[kernelnum.length-1]; k++){
			n=p;
			for(int i=0; i<pooloutsize[pooloutsize.length-1][0]; i++){
				if(i!=0){
					n=i*z[k].rows+p;
				}
				f.putColumn(n, z[k].getRow(i));
			}
			if(k+1<kernelnum[kernelnum.length-1]){
				p=n+z[k+1].rows;
			}
		}
		return f;
	}


	public FloatMatrix[] data_unflat(FloatMatrix x){
		FloatMatrix[] z = new FloatMatrix[kernelnum[kernelnum.length-1]];
		int n=0;

		for(int k=0; k<z.length; k++){
			z[k] = new FloatMatrix(pooloutsize[pooloutsize.length-1][0],pooloutsize[pooloutsize.length-1][1]);
			for(int i=0; i<z[k].rows; i++)
				for(int j=0; j<z[k].columns; j++){
					z[k].put(i, j, x.get(n));
					n++;
				}
		}

		return z;
	}


}
