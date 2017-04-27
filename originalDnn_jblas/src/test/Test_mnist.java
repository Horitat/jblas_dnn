package test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import layers.Convolutionlayer;
import layers.Hiddenlayer;
import layers.MaxPoolinglayer;
import read_inputdata.Read_img;
import read_inputdata.Read_label;
import read_inputdata.Read_number_of_data;
import single_layer.Logistic_kaiki;
import util.Common_method;
import Mersenne.Sfmt;


public class Test_mnist {

	int[] kernelnum;
	int[][] convkernelsize;
	int[][] poolkernelsize;
	int[] hidden_N;
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
	Hiddenlayer[] hidden;
	Logistic_kaiki logistic;

	Sfmt mt;

	//LeNetを構築
	public Test_mnist(int[] imageSize, int channel, int[] nKernels, int[][] kernelSizes, int[][] poolSizes,
			int[] stride, int[] nHidden, int nOut,int minibatch, Sfmt m, String activation){

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
			int o = (Common_method.compute_outputN(size[0],convkernelsize[i][0],1,0));
			convoutsize[i] = new int[]{o,o};
			o = Common_method.compute_outputN(convoutsize[i][0],poolSizes[i][0],1,0);
			pooloutsize[i] = new int[]{o,o};
			//			convoutsize[i] = new int[]{size[0] - kernelSizes[i][0] + 1, size[1] - kernelSizes[i][1] + 1};
			//			pooloutsize[i] = new int[]{convoutsize[i][0] / poolSizes[i][0], convoutsize[i][1] / poolSizes[i][1]};
			//			System.out.println("convout:"+ (size[0] - kernelSizes[i][0] + 1)+","+ (size[1] - kernelSizes[i][1] + 1));
			//			System.out.println("poolout:"+ convoutsize[i][0] / poolSizes[i][0] +","+ convoutsize[i][1] / poolSizes[i][1]);

			conv[i] = new Convolutionlayer(size,chnl, nKernels[i], kernelSizes[i],convoutsize[i],stride[i*2], minibatch, mt, activation);
			pool[i] = new MaxPoolinglayer(poolkernelsize[i],pooloutsize[i], nKernels[i], stride[i*2+1], minibatch,mt, "MAX", "");
		}

		//入力データを一次元に直すためのサイズ、全結合層への入力に使われる
		flatsize = nKernels[nKernels.length-1] * pooloutsize[pooloutsize.length-1][0]* pooloutsize[pooloutsize.length-1][1];
		System.out.println("flatsize:"+nKernels[nKernels.length-1] +":"+ pooloutsize[pooloutsize.length-1][0] +":"+ pooloutsize[pooloutsize.length-1][1]);

		hidden = new Hiddenlayer[hidden_N.length];
		for(int i=0; i< hidden_N.length; i++){
			hidden[i] = new Hiddenlayer(flatsize, hidden_N[i], null, null, minibatch, mt, activation);
		}
		logistic = new Logistic_kaiki(hidden_N[hidden_N.length-1], output_N);
	}


	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ

		//時間計測
		long start = System.currentTimeMillis();

		System.out.println("Load input data....");
		int train_N = Read_number_of_data.count_data("C:\\pleiades\\originaldnn_test_data\\mnistdata\\mnist_train.txt");
		int test_N = Read_number_of_data.count_data("C:\\pleiades\\originaldnn_test_data\\mnistdata\\mnist_test.txt");

		float[][][][] train_data = Read_img.read_color_img_to_gray("C:\\pleiades\\originaldnn_test_data\\mnistdata\\train\\", 0, 0);
		int[][] train_label = Read_label.Classifier_label("C:\\pleiades\\originaldnn_test_data\\mnistdata\\mnist_train.txt", train_N);

		long end = System.currentTimeMillis();
		System.out.println("time of load input train data:" + (end - start)  + "ms");

		float[][][][] test_data = Read_img.read_color_img_to_gray("C:\\pleiades\\originaldnn_test_data\\mnistdata\\test\\", 0, 0);
		int[][] test_label = Read_label.Classifier_label("C:\\pleiades\\originaldnn_test_data\\mnistdata\\mnist_test.txt", train_N);

		end = System.currentTimeMillis();
		System.out.println("time of load input test data:" + (end - start) / 1000 + "second");

		//System.out.println(train_data[0][0].length+":"+ train_data[0][0][0].length);
		int[] inputsize = {train_data[0][0].length, train_data[0][0][0].length};
		int chanel = train_data[0].length;
		int minibatchsize = 1000;
		int minibatch_N = train_N / minibatchsize;
		int output_N = train_label[0].length;

		float l_rate = 0.01f;

		float[][][][][] train_data_minibatch = new float[minibatch_N][minibatchsize][chanel][inputsize[0]][inputsize[1]];
		int[][][] train_label_minibatch = new int[minibatch_N][minibatchsize][output_N];
		List<Integer> minibatchIndex = new ArrayList<>();

		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		Sfmt mt = new Sfmt(init_key);

		for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
		Common_method.list_shuffle(minibatchIndex, mt);

		// create minibatches
		for (int j = 0; j < minibatchsize; j++) {
			for (int i = 0; i < minibatch_N; i++) {
				train_data_minibatch[i][j] = train_data[minibatchIndex.get(i * minibatchsize + j)];
				train_label_minibatch[i][j] = train_label[minibatchIndex.get(i * minibatchsize + j)];
			}
		}
		System.out.println("...done");

		final int channel = 1;

		int[] nKernels = {20, 50};//カーネルの数
		int[][] kernelSize = { {5, 5}, {5, 5} };//カーネルサイズ
		int[][] poolSize = { {2, 2}, {2, 2} };//プーリングのカーネルサイズ
		int[] nHidden = {800, 500};
		int[] stride = {1,1,1,2};

		System.out.println("Building the LeNet model...");
		Test_mnist Lenet = new Test_mnist(inputsize, channel, nKernels, kernelSize, poolSize, stride, nHidden, output_N,minibatch_N, mt, "ReLU");
		System.out.println("done.");

		System.out.println("Training the model...");
		System.out.println();
		int epochs = 500;
		for (int epoch = 0; epoch < epochs; epoch++) {

			if ((epoch + 1) % 50 == 0) {
				System.out.println("\titer = " + (epoch + 1) + " / " + epochs);
			}
//			System.out.println("\titer = " + (epoch + 1) + " / " + epochs);


			for (int batch = 0; batch < minibatch_N; batch++) {
				Lenet.train(train_data_minibatch[batch], train_label_minibatch[batch], minibatchsize, l_rate);
//				end = System.currentTimeMillis();
//				System.out.println("one of end minibatch train("+batch+"):" + (end - start) / 1000 + "second");
			}
			l_rate *= 0.999;
		}
		System.out.println("done.");

		end = System.currentTimeMillis();
		System.out.println("time of end NN train:" + (end - start) / 1000 + "second");

		Integer[][] predicted_T = new Integer[test_N][output_N];
		// test
		for (int i = 0; i < test_N; i++) {
			predicted_T[i] = Lenet.predict(test_data[i]);
		}

		int[][] confusionMatrix = new int[output_N][output_N];
		double accuracy = 0.;
		double[] precision = new double[output_N];
		double[] recall = new double[output_N];

		for (int i = 0; i < test_N; i++) {
			int predicted_ = Arrays.asList(predicted_T[i]).indexOf(1);
			int actual_ = Arrays.asList(test_label[i]).indexOf(1);

			confusionMatrix[actual_][predicted_] += 1;
		}

		for (int i = 0; i < output_N; i++) {
			double col_ = 0.;
			double row_ = 0.;

			for (int j = 0; j < output_N; j++) {

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
		for (int i = 0; i < output_N; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, precision[i] * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < output_N; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, recall[i] * 100);
		}

		end = System.currentTimeMillis();
		System.out.println("time of end this program:" + (end - start) / 1000 + "second");

	}


	/**
	 *
	 * @param x [minibatchsize][chanel][imgsize][imgsize]
	 * @param label
	 * @param minibatchsize
	 * @param l_rate
	 */
	public void train(float[][][][] x, int[][] label, int minibatchsize, float l_rate){
		//各層のインプットデータ,活性化後をキャッシュ
		List<float[][][][]> preactdata = new ArrayList<>(kernelnum.length);
		List<float[][][][]> after_act = new ArrayList<>(kernelnum.length);
		//入力データを保持のための+1,プーリングの結果を保持
		List<float[][][][]> downsampling = new ArrayList<>(kernelnum.length+1);
		downsampling.add(x);

		//初期化
		for(int i=0; i<kernelnum.length; i++){
			//convで代入しconvのbackwardで使用
			preactdata.add(new float[minibatchsize][kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
			//convで代入しpoolingのbackwordで使用
			after_act.add(new float[minibatchsize][kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
			//
			downsampling.add(new float[minibatchsize][kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
		}

		//一次元に変換したデータをキャッシュ用
		float[][] flatdata = new float[minibatchsize][flatsize];
		//隠れ層の出力キャッシュ用
		float[][][] hiddendata = new float[hidden_N.length][minibatchsize][];

		for(int i=0; i<hidden_N.length; i++){
			for(int m=0; m<minibatchsize; m++){
				hiddendata[i][m] = new float[hidden_N[i]];
				//System.out.println("length"+hiddendata[i][m].length);
			}
		}


		//出力層の逆伝播
		float[][] dy;
		//隠れ層の逆伝播
		float[][] dz;
		//一次元変換の逆伝播
		float[][] dx_flat = new float[minibatchsize][flatsize];
		//畳込み、プーリングの逆伝播
		float[][][][] dx = new float[minibatchsize][kernelnum[kernelnum.length-1]][pooloutsize[pooloutsize.length-1][0]][pooloutsize[pooloutsize.length-1][1]];

		float[][][][] dc;

		//batchnormがありかなしかで条件分けし、ミニバッチでループか各層をミニバッチ単位でループするか
		for(int n=0; n<minibatchsize; n++){
			//順伝播
			float[][][] z = x[n].clone();
			for(int i=0; i<kernelnum.length; i++){
				z = conv[i].forward(z, preactdata.get(i)[n], after_act.get(i)[n]);
				z = pool[i].maxpooling(z);

				downsampling.get(i+1)[n] = z.clone();
			}

			float[] xx = data_flat(z);
			flatdata[n] = xx.clone();
			for(int h=0; h<hidden.length; h++){
				if(h == 0){
					hiddendata[h][n] = hidden[h].forward(xx);
				}else{
					hiddendata[h][n] = hidden[h].forward(hiddendata[h-1][n]);
				}
			}
		}
		//出力層の順伝播
		dy = logistic.train(hiddendata[hiddendata.length-1], label, minibatchsize, l_rate);
		//全結合層の逆伝播
		//		dz = hidden.backward(flatdata, hiddendata, dy, logistic.weight, minibatchsize, l_rate);
		dz = dy.clone();
		for(int h = hidden.length-1; h >= 0; h--){
			if(h == hidden.length -1){
				dz = hidden[h].backward(flatdata, hiddendata[h], dz, logistic.weight, minibatchsize, l_rate);
			}else{
				dz = hidden[h].backward(flatdata, hiddendata[h], dz, hidden[h+1].weight, minibatchsize, l_rate);
			}
		}
		//System.out.println("end all connect layer backward");
		//畳込み層、プーリング層の逆伝播のために、フラット化したデータを戻す
		for(int n=0; n<minibatchsize; n++){
			for(int i=0; i<flatsize; i++)
				for(int j=0; j<hidden_N[0]; j++){
					dx_flat[n][i] += hidden[0].weight[j][i] * dz[n][j];
				}
			dx[n] = data_unflat(dx_flat[n]);
		}
		//System.out.println("start pooling and conv layer backward");
		//畳込み層の逆伝播
		dc = dx.clone();
		for(int i= kernelnum.length-1; i>-1; i--){
			float[][][][] poolback = pool[i].backmaxpooing( after_act.get(i), downsampling.get(i+1), dc, convoutsize[i], minibatchsize);
			dc = conv[i].backward(downsampling.get(i), preactdata.get(i), poolback, minibatchsize, l_rate);
		}

		//System.out.println("end 1 minibatch");
	}


	private Integer[] predict(float[][][] x) {
		// TODO 自動生成されたメソッド・スタブ
		List<float[][][]> preact = new ArrayList<>(kernelnum.length);
		List<float[][][]> act = new ArrayList<>(kernelnum.length);

		for(int i=0; i<kernelnum.length; i++){
			preact.add(new float[kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
			act.add(new float[kernelnum[i]][convoutsize[i][0]][convoutsize[i][1]]);
		}

		float[][][] z = x.clone();

		for(int i=0; i<kernelnum.length; i++){
			z = conv[i].forward(z, preact.get(i), act.get(i));
			z = pool[i].maxpooling(z);
		}

		float[] data = data_flat(z);
		for(int h=0; h<hidden.length; h++){
			data = hidden[h].forward(data);
		}

		return logistic.predict(data);
	}


	private float[] data_flat(float[][][] z) {
		// TODO 自動生成されたメソッド・スタブ
		float[] f = new float[flatsize];
		//nKernels[nKernels.length-1] +":"+ pooloutsize[pooloutsize.length-1][0] +":"+ pooloutsize[pooloutsize.length-1][1]
		//System.out.println(z.length+":"+z[0].length+":"+z[0][0].length);
		int n=0;
		for(int k=0; k<kernelnum[kernelnum.length-1]; k++)
			for(int i=0; i<pooloutsize[pooloutsize.length-1][0]; i++)
				for(int j=0; j<pooloutsize[pooloutsize.length-1][1]; j++){
					f[n] = z[k][i][j];
					n++;
				}

		return f;
	}


	public float[][][] data_unflat(float[] x){
		float[][][] z = new float[kernelnum[kernelnum.length-1]][pooloutsize[pooloutsize.length-1][0]][pooloutsize[pooloutsize.length-1][1]];
		int n=0;

		for(int k=0; k<z.length; k++)
			for(int i=0; i<z[0].length; i++)
				for(int j=0; j<z[0][0].length; j++){
					z[k][i][j] = x[n];
					n++;
				}

		return z;
	}


}
