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

public class CNN {

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

	public CNN(int[] imageSize, int channel, int[] nKernels, int minibatch,
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
//			pooloutsize[i] = new int[]{convoutsize[i][0] / poolSizes[i][0], convoutsize[i][1] / poolSizes[i][1]};
			pooloutsize[i] = new int[]{(((convoutsize[i][0]-poolSizes[i][0]+(2*0)) / 1)+1), (((convoutsize[i][1]-poolSizes[i][1]+(2*0)) / 1)+1)};
			System.out.println("convout:"+ (size[0] - kernelSizes[i][0] + 1)+","+ (size[1] - kernelSizes[i][1] + 1));
//			System.out.println("poolout:"+ convoutsize[i][0] / poolSizes[i][0] +","+ convoutsize[i][1] / poolSizes[i][1]);
			System.out.println("poolout1:"+ (((convoutsize[i][0]-poolSizes[i][0]+(2*0)) / 1)+1) +","+ (((convoutsize[i][1]-poolSizes[i][1]+(2*0)) / 1)+1));

			conv[i] = new Convolutionlayer(size,chnl, nKernels[i], kernelSizes[i],convoutsize[i], 1,minibatch, mt, activation);
			pool[i] = new MaxPoolinglayer(poolkernelsize[i],pooloutsize[i], nKernels[i], 1, minibatch,mt, "MAX", "");
		}

		//入力データを一次元に直すためのサイズ、全結合層への入力に使われる
		flatsize = nKernels[nKernels.length-1] * pooloutsize[pooloutsize.length-1][0]* pooloutsize[pooloutsize.length-1][1];
		System.out.println("flatsize:"+nKernels[nKernels.length-1] +":"+ pooloutsize[pooloutsize.length-1][0] +":"+ pooloutsize[pooloutsize.length-1][1]);

		hidden = new Hiddenlayer(flatsize, hidden_N, null, null, minibatch,mt, activation);
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

		FloatMatrix[][] train_X = new FloatMatrix[train_N][channel];//[imageSize[0]][imageSize[1]];
		FloatMatrix train_T = new FloatMatrix(train_N,nOut);

		FloatMatrix[][] test_X = new FloatMatrix[test_N][channel];//[imageSize[0]][imageSize[1]];
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
		FloatMatrix[][][] train_X_minibatch = new FloatMatrix[minibatch_N][minibatchSize][channel];//[imageSize[0]][imageSize[1]];
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
		CNN classifier = new CNN(imageSize, channel, nKernels,minibatchSize, kernelSizes, poolSizes, nHidden, nOut, mt, "ReLU");
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
		System.out.println("jblas DNN end");

/*		int[][] confusionMatrix = new int[patterns][patterns];
		double accuracy = 0.;
		double[] precision = new double[patterns];
		double[] recall = new double[patterns];

		for (int i = 0; i < test_N; i++) {
			int predicted_ = Arrays.asList(predicted_T[i]).indexOf(1);
			int actual_ = Arrays.asList(test_T[i]).indexOf(1);

			confusionMatrix[actual_][predicted_] += 1;
		}

		for (int i = 0; i < patterns; i++) {
			double col_ = 0.;
			double row_ = 0.;

			for (int j = 0; j < patterns; j++) {

				if (i == j) {
					accuracy += confusionMatrix[i][j];
					precision[i] += confusionMatrix[j][i];
					recall[i] += confusionMatrix[i][j];
				}

				col_ += confusionMatrix[j][i];
				row_ += confusionMatrix[i][j];
			}
			precision[i] /= col_;
			recall[i] /= row_;
		}

		accuracy /= test_N;

		System.out.println("--------------------");
		System.out.println("CNN model evaluation");
		System.out.println("--------------------");
		System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, precision[i] * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, recall[i] * 100);
		}*/
	}



	/**
	 *
	 * @param x [minibatchsize][chanel][imgsize][imgsize]
	 * @param label
	 * @param minibatchsize
	 * @param l_rate
	 */
	public void train(FloatMatrix[][] x, FloatMatrix label, int minibatchsize, float l_rate){
//		public void train(float[][][][] x, int[][] label, int minibatchsize, float l_rate){
		//各層のインプットデータ,活性化後をキャッシュ
		List<FloatMatrix[][]> preactdata = new ArrayList<>(kernelnum.length);
		List<FloatMatrix[][]> after_act = new ArrayList<>(kernelnum.length);
		//入力データを保持のための+1,プーリングの結果を保持
		List<FloatMatrix[][]> downsampling = new ArrayList<>(kernelnum.length+1);
//		List<float[][][][]> preactdata = new ArrayList<>(kernelnum.length);
//		List<float[][][][]> after_act = new ArrayList<>(kernelnum.length);
		//入力データを保持のための+1,プーリングの結果を保持
//		List<float[][][][]> downsampling = new ArrayList<>(kernelnum.length+1);
		downsampling.add(x);

		//初期化
//		for(int i=0; i<kernelnum.length; i++){
			//convで代入しconvのbackwardで使用
//			preactdata.add(new FloatMatrix[minibatchsize][kernelnum[i]]);
			//convで代入しpoolingのbackwordで使用
//			after_act.add(new FloatMatrix[minibatchsize][kernelnum[i]]);
//			downsampling.add(new FloatMatrix[minibatchsize][kernelnum[i]]);
//			preactdata.add(new float[minibatchsize][kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
//			after_act.add(new float[minibatchsize][kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
//			downsampling.add(new float[minibatchsize][kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
//		}


		for(int i=0; i<kernelnum.length;i++){
			FloatMatrix[][] init_preact = new FloatMatrix[minibatchsize][kernelnum[i]];
			FloatMatrix[][] init_act = new FloatMatrix[minibatchsize][kernelnum[i]];
			FloatMatrix[][] init_down = new FloatMatrix[minibatchsize][kernelnum[i]];
			for(int n=0; n<minibatchsize; n++){
//				init = new FloatMatrix[minibatchsize][kernelnum[i]];
				for(int j=0; j<kernelnum[i]; j++){
					init_preact[n][j] = new FloatMatrix(convoutsize[i][0],convoutsize[i][1]);
					init_act[n][j] = new FloatMatrix(convoutsize[i][0],convoutsize[i][1]);
					init_down[n][j] = new FloatMatrix(convoutsize[i][0],convoutsize[i][1]);
				}
			}

			preactdata.add(init_preact.clone());
			//convで代入しpoolingのbackwordで使用
			after_act.add(init_act.clone());
			downsampling.add(init_down.clone());
//			System.out.println(convoutsize[i][0]+"*********"+convoutsize[i][1]);
		}

		//一次元に変換したデータをキャッシュ用
		FloatMatrix flatdata = new FloatMatrix(minibatchsize,flatsize);
		//隠れ層の出力キャッシュ用
		FloatMatrix hiddendata = new FloatMatrix(minibatchsize,hidden_N);

/*		float[][] flatdata = new float[minibatchsize][flatsize];
		//隠れ層の出力キャッシュ用
		float[][] hiddendata = new float[minibatchsize][hidden_N];
		//出力層の逆伝播
		float[][] dy;
		//隠れ層の逆伝播
		float[][] dz;
		//一次元変換の逆伝播
		float[][] dx_flat = new float[minibatchsize][flatsize];
		//畳込み、プーリングの逆伝播
		float[][][][] dx = new float[minibatchsize][kernelnum[kernelnum.length-1]][pooloutsize[pooloutsize.length-1][0]][pooloutsize[pooloutsize.length-1][1]];

		float[][][][] dc;
*/

//		for(int i=0; i<x.length; i++){
//			for(int j=0; j<x[i].length; j++){
//				System.out.println(x[i][j]);
//			}
//		}

		//batchnormがありかなしかで条件分けし、ミニバッチでループか各層をミニバッチ単位でループするか
		for(int n=0; n<minibatchsize; n++){
		//順伝播x[minibatchSize][channel]
			FloatMatrix[] z = x[n].clone();
//			float[][][] z = x[n].clone();
			for(int i=0; i<kernelnum.length; i++){
//				System.out.println(z[i]);
//				System.out.println("???"+after_act.get(i)[0][0]);
				z = conv[i].forward(z, preactdata.get(i)[n], after_act.get(i)[n]);
//				System.out.println(z[i]+"\n///");
				//				System.out.println(z[0]);
//				scan.next();
				z = pool[i].maxpooling(z);
//				System.out.println(z[i]+"\n---");
				//System.out.println("kernel train");
//				System.out.println(z[0].rows+"++"+z[0].columns);

				downsampling.get(i+1)[n] = z.clone();
			}

			FloatMatrix xx = data_flat(z);
			flatdata.putRow(n, xx.dup());
			hiddendata.putRow(n, hidden.forward(xx.dup()));
//			System.out.println(hiddendata+"\n*");
//			hiddendata = hidden.forward(xx.dup());
			//System.out.println("minibatch:"+n);
//			scan.nextLine();
		}
//		scan.nextLine();
//		System.out.println("##################################");
//		for(int i=0; i<kernelnum.length; i++){
//			for(int n=0; n<minibatchsize; n++){
//				for(int j=0; j<kernelnum[i]; j++){
//					System.out.println(after_act.get(i)[n][j]);
//				}
//			}
//		}

		//出力層の逆伝播
		FloatMatrix dx_flat = new FloatMatrix(minibatchsize,flatsize);
		//出力層の順伝播逆伝播
//		System.out.println(hiddendata.rows+"*"+hiddendata.columns);
//		System.out.println(label.rows+"^"+label.columns);
//		System.out.println("*"+minibatchsize);
		System.out.println(hiddendata+"\n*");
//		System.out.println("*"+minibatchsize);
		FloatMatrix dy = logistic.train(hiddendata, label, minibatchsize, l_rate);
//		System.out.println(dy+"\n++");
		//隠れ層の逆伝播
		FloatMatrix dz = hidden.backward(flatdata, hiddendata, dy, logistic.weight, minibatchsize, l_rate);
scan.nextLine();
		//出力層の
//		dy = logistic.train(hiddendata, label, minibatchsize, l_rate);
		//全結合層の逆伝播
//		dz = hidden.backward(flatdata, hiddendata, dy, logistic.weight, minibatchsize, l_rate);

		//畳込み、プーリングの逆伝播
		FloatMatrix[][] dx = new FloatMatrix[minibatchsize][kernelnum[kernelnum.length-1]];//[pooloutsize[pooloutsize.length-1][0]][pooloutsize[pooloutsize.length-1][1]];

		//畳込み層、プーリング層の逆伝播のために、フラット化したデータを戻す
		dx_flat = dz.mmul(hidden.weight);
		for(int n=0; n<minibatchsize; n++){
//			for(int i=0; i<flatsize; i++){
//				for(int j=0; j<hidden_N; j++){
//					dx_flat[n][i] += hidden.weight[j][i] * dz[n][j];
//				}
//			}
			dx[n] = data_unflat(dx_flat.getRow(n));
		}

		//畳込み層の逆伝播
		FloatMatrix[][] dc = dx.clone();
		for(int i= kernelnum.length-1; i>-1; i--){
//			System.out.println("!!!"+after_act.get(i)[0][0]);
			FloatMatrix[][] poolback = pool[i].backmaxpooing( after_act.get(i), downsampling.get(i+1), dc, convoutsize[i], minibatchsize);
			dc = conv[i].backward(downsampling.get(i), preactdata.get(i), poolback, minibatchsize, l_rate);
		}
//		for(int i= kernelnum.length-1; i>-1; i--){
//			float[][][][] poolback = pool[i].backmaxpooing( after_act.get(i), downsampling.get(i+1), dc, convoutsize[i], minibatchsize);
//			dc = conv[i].backward(downsampling.get(i), preactdata.get(i), poolback, minibatchsize, l_rate);
//		}
	}


	private FloatMatrix predict(FloatMatrix[] x) {
//		private Integer[] predict(float[][][] x) {
		// TODO 自動生成されたメソッド・スタブ
		List<FloatMatrix[]> preact = new ArrayList<>(kernelnum.length);
		List<FloatMatrix[]> act = new ArrayList<>(kernelnum.length);

//		for(int i=0; i<kernelnum.length; i++){
//			preact.add(new FloatMatrix[kernelnum[i]]);//[convoutsize[i][0]][convoutsize[i][1]]);
//			act.add(new FloatMatrix[kernelnum[i]]);//[convoutsize[i][0]][convoutsize[i][1]]);
//		}

		for(int i=0; i<kernelnum.length;i++){
			FloatMatrix[] init = new FloatMatrix[kernelnum[i]];
			for(int j=0; j<kernelnum[i]; j++){
					init[j] = new FloatMatrix(convoutsize[i][0],convoutsize[i][1]);
			}

			preact.add(init.clone());
			//convで代入しpoolingのbackwordで使用
			act.add(init.clone());
//			System.out.println(convoutsize[i][0]+"*********"+convoutsize[i][1]);
		}

		FloatMatrix[] z = x.clone();

		for(int i=0; i<kernelnum.length; i++){
			z = conv[i].forward(z, preact.get(i), act.get(i));
			z = pool[i].maxpooling(z);
		}
		return logistic.predict(hidden.forward(data_flat(z)));
	}


	private FloatMatrix data_flat(FloatMatrix[] z) {
//		private float[] data_flat(float[][][] z) {
		// TODO 自動生成されたメソッド・スタブ
		FloatMatrix f = new FloatMatrix(1,flatsize);
//		float[] f = new float[flatsize];
		//nKernels[nKernels.length-1] +":"+ pooloutsize[pooloutsize.length-1][0] +":"+ pooloutsize[pooloutsize.length-1][1]
		//System.out.println(z.length+":"+z[0].length+":"+z[0][0].length);
//		System.out.println(f.length+"WW"+z.length);
		//z[channel] (outputsize,outputsize)
		int n=0,p=0;
		for(int k=0; k<kernelnum[kernelnum.length-1]; k++){
			n=p;
			for(int i=0; i<pooloutsize[pooloutsize.length-1][0]; i++){
				if(i!=0){
					n=i*z[k].rows+p;
				}
//				System.out.println(z[k]);
//				System.out.println(i+"L"+n);
				f.putColumn(n, z[k].getRow(i));
			}
			if(k+1<kernelnum[kernelnum.length-1]){
				p=n+z[k+1].rows;
			}
//				for(int j=0; j<pooloutsize[pooloutsize.length-1][1]; j++){
//					f[n] = z[k][i][j];
//					n++;
//				}
		}
		return f;
	}


	public FloatMatrix[] data_unflat(FloatMatrix x){
		FloatMatrix[] z = new FloatMatrix[kernelnum[kernelnum.length-1]];//[pooloutsize[pooloutsize.length-1][0]][pooloutsize[pooloutsize.length-1][1]];
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
