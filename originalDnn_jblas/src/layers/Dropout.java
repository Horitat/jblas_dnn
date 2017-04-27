package layers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.jblas.FloatMatrix;

import single_layer.Logistic_kaiki;
import util.Common_method;
import util.RandomGenerator;
import Mersenne.Sfmt;

public class Dropout {

	static float pdrop = 0.5f;
	int input_N;
	int output_N;
	Sfmt mt;
	int[] hidden_N;//各隠れ層のノード数
	int hiddenlayer_N;//隠れ層の数
	Hiddenlayer[] hiddenlayer;
	Logistic_kaiki logisticlayer;

	public Dropout(int input, int[] hidden, int output,int minibatch, Sfmt m, String activation){
		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}
		if(StringUtils.isEmpty(activation)){
			activation = "ReLU";
		}

		input_N = input;
		hidden_N = hidden;
		output_N = output;
		hiddenlayer_N = hidden.length;
		mt = m;
		hiddenlayer = new Hiddenlayer[hiddenlayer_N];
		System.out.println("hidden length:"+hidden.length);

		for(int i=0; i< hiddenlayer_N; i++){
			int in;
			if(i == 0){
				in = input;
			}else{
				in = hidden[i-1];
			}
			hiddenlayer[i] = new Hiddenlayer(in, hidden_N[i], null, null,minibatch, mt, activation);
			System.out.println("hidden node:"+hidden_N[i]);
		}
		logisticlayer = new Logistic_kaiki(hidden[hiddenlayer_N-1],output);

	}


	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		Sfmt mt = new Sfmt(init_key);
		//
		// Declare variables and constants
		//
		int train_N_each = 300;        // for demo
		int test_N_each = 50;          // for demo
		int nIn_each = 20;             // for demo
		double pNoise_Training = 0.2;  // for demo
		double pNoise_Test = 0.25;     // for demo

		final int patterns = 3;

		final int train_N = train_N_each * patterns;
		final int test_N = test_N_each * patterns;

		final int nIn = nIn_each * patterns;
		final int nOut = patterns;

		int[] hiddenLayerSizes = {10, 80};
		//double pDropout = 0.5;

		FloatMatrix train_X = new FloatMatrix(train_N,nIn);
		FloatMatrix train_T = new FloatMatrix(train_N,nOut);
//		float[][] train_X = new float[train_N][nIn];
//		int[][] train_T = new int[train_N][nOut];

		FloatMatrix test_X = new FloatMatrix(test_N,nIn);
		FloatMatrix test_T = new FloatMatrix(test_N,nOut);
//		float[][] test_X = new float[test_N][nIn];
//		Integer[][] test_T = new Integer[test_N][nOut];



		int epochs = 5000 / 10;
		float learningRate = 0.2f;

		int minibatchSize = 50;
		final int minibatch_N = train_N / minibatchSize;

		//
		// Create training data and test data for demo.
		//
		for (int pattern = 0; pattern < patterns; pattern++) {

			for (int n = 0; n < train_N_each; n++) {

				int n_ = pattern * train_N_each + n;

				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= train_N_each * pattern && n_ < train_N_each * (pattern + 1) ) &&
							(i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
						train_X.put(n_,i,(float) (RandomGenerator.binomial(1, 1 - pNoise_Training, mt) * mt.NextUnif() * .5 + .5));
//						train_X[n_][i] = (float) (RandomGenerator.binomial(1, 1 - pNoise_Training, mt) * mt.NextUnif() * .5 + .5);
					} else {
						train_X.put(n_,i,(float) (RandomGenerator.binomial(1, pNoise_Training, mt) * mt.NextUnif() * .5 + .5));
//						train_X[n_][i] = (float) (RandomGenerator.binomial(1, pNoise_Training, mt) * mt.NextUnif() * .5 + .5);
					}
				}

				for (int i = 0; i < nOut; i++) {
					if (i == pattern) {
						train_T.put(n_,i,1);
//						train_T[n_][i] = 1;
					} else {
						train_T.put(n_,i,0);
//						train_T[n_][i] = 0;
					}
				}
			}

			for (int n = 0; n < test_N_each; n++) {

				int n_ = pattern * test_N_each + n;

				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1) ) &&
							(i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
						test_X.put(n_,i,(float) ( RandomGenerator.binomial(1, 1 - pNoise_Test, mt) * mt.NextUnif() * .5 + .5));
//						test_X[n_][i] = (float) ( RandomGenerator.binomial(1, 1 - pNoise_Test, mt) * mt.NextUnif() * .5 + .5);
					} else {
						test_X.put(n_,i,(float) ( RandomGenerator.binomial(1, pNoise_Test, mt) * mt.NextUnif() * .5 + .5));
//						test_X[n_][i] = (float) ( RandomGenerator.binomial(1, pNoise_Test, mt) * mt.NextUnif() * .5 + .5);
					}
				}

				for (int i = 0; i < nOut; i++) {
					if (i == pattern) {
						test_T.put(n_,i,1);
//						test_T[n_][i] = 1;
					} else {
						test_T.put(n_,i,0);
//						test_T[n_][i] = 0;
					}
				}
			}
		}



		FloatMatrix[] train_X_minibatch = new FloatMatrix[minibatch_N];//[minibatchSize][nIn];
		FloatMatrix[] train_T_minibatch = new FloatMatrix[minibatch_N];//[minibatchSize][nOut];
//		float[][][] train_X_minibatch = new float[minibatch_N][minibatchSize][nIn];
//		int[][][] train_T_minibatch = new int[minibatch_N][minibatchSize][nOut];

		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < train_N; i++) minibatchIndex.add(i);
		Common_method.list_shuffle(minibatchIndex, mt);

		// create minibatches
		for (int i = 0; i < minibatch_N; i++) {
			train_X_minibatch[i] = new FloatMatrix(minibatchSize,nIn);
			train_T_minibatch[i] = new FloatMatrix(minibatchSize,nOut);
			for (int j = 0; j < minibatchSize; j++) {
				train_X_minibatch[i].putRow(j, train_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
				train_T_minibatch[i].putRow(j, train_T.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
		}
//		for (int j = 0; j < minibatchSize; j++) {
//			for (int i = 0; i < minibatch_N; i++) {
//				train_X_minibatch[i][j] = train_X[minibatchIndex.get(i * minibatchSize + j)];
//				train_T_minibatch[i][j] = train_T[minibatchIndex.get(i * minibatchSize + j)];
//			}
//		}

		//
		// Build Dropout model
		//

		// construct Dropout
		System.out.println("Building the model...");
		Dropout classifier = new Dropout(nIn, hiddenLayerSizes, nOut, minibatchSize,mt, "ReLU");
		System.out.println("done.");

		// train the model
		System.out.println("Training the model...");
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int batch = 0; batch < minibatch_N; batch++) {
				classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate);
			}
			learningRate *= 0.999;
		}
		System.out.println("done.");

		// adjust the weight for testing
		System.out.println("Optimizing weights before testing...");
		classifier.adjust_weight();
		System.out.println("done.");


		// test
//		for (int i = 0; i < test_N; i++) {
//			predicted_T[i] = classifier.predict(test_X[i]);
//		}

		FloatMatrix predicted_T = classifier.predict(test_X);
//		FloatMatrix predicted_T = new FloatMatrix(test_N,nOut);
//		Integer[][] predicted_T = new Integer[test_N][nOut];
		//
		// Evaluate the model
		//
		double accuracy = 0.;

		FloatMatrix confusionMatrix = new FloatMatrix(patterns,patterns);
		FloatMatrix precision = new FloatMatrix(patterns);
		FloatMatrix recall = new FloatMatrix(patterns);
//		int[][] confusionMatrix = new int[patterns][patterns];
//		double[] precision = new double[patterns];
//		double[] recall = new double[patterns];

		for (int i = 0; i < test_N; i++) {
			Integer[] predict_array = new Integer[predicted_T.getRow(i).toIntArray().length];
			Integer[] label_array = new Integer[test_T.getRow(i).toIntArray().length];

			for(int j=0; j<predict_array.length; j++){
				predict_array[j] = predicted_T.getRow(i).toIntArray()[j];
				label_array[j] =test_T.getRow(i).toIntArray()[j];
//				System.out.println(j);
			}

			int predicted_ = Arrays.asList(predict_array).indexOf(1);
			int actual_ = Arrays.asList(label_array).indexOf(1);

			if(predicted_ != -1 && actual_ != -1){
//				confusionMatrix[actual_][predicted_] += 1;
			}
			confusionMatrix.put(predicted_,actual_, confusionMatrix.get(predicted_,actual_)+1f);
		}

		for(int i=0; i<patterns; i++){
			accuracy += confusionMatrix.get(i,i);
			precision.put(i, (precision.get(i)+confusionMatrix.get(i,i))/confusionMatrix.getColumn(i).sum());
			recall.put(i, (recall.get(i)+confusionMatrix.get(i,i))/confusionMatrix.getRow(i).sum());
		}

//		for (int i = 0; i < patterns; i++) {
//			double col_ = 0.;
//			double row_ = 0.;
//
//			for (int j = 0; j < patterns; j++) {
//				if (i == j) {
//					accuracy += confusionMatrix[i][j];
//					precision[i] += confusionMatrix[j][i];
//					recall[i] += confusionMatrix[i][j];
//				}
//
//				col_ += confusionMatrix[j][i];
//				row_ += confusionMatrix[i][j];
//			}
//			precision[i] /= col_;
//			recall[i] /= row_;
//		}

		accuracy /= test_N;

		System.out.println("------------------------");
		System.out.println("Dropout model evaluation");
		System.out.println("------------------------");
		System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, precision.get(i) * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, recall.get(i) * 100);
		}
	}



	public void train(FloatMatrix x, FloatMatrix label, int minibatchSize, float l_rate){
//		public void train(float[][] x, int[][] label, int minibatchSize, float l_rate){

//		List<float[][]> inputdata = new ArrayList<>(hiddenlayer_N+1);

//		List<int[][]> dropoutmask = new ArrayList<>(hiddenlayer_N);
		//		float[][] z= new float[0][0]; //各層の出力値を格納(上書きし続ける)、ロジスティック層の値を格納
		List<FloatMatrix> dropoutmask = new ArrayList<>(hiddenlayer_N);
		FloatMatrix z = x.dup(); //各層の出力値を格納、ロジスティック層の値を格納

		List<FloatMatrix> inputdata = new ArrayList<>(hiddenlayer_N+1);
		inputdata.add(x);
		//順伝播
		for(int layer=0; layer<hiddenlayer_N; layer++){
//			z = hiddenlayer[layer].forward(inputdata.get(inputdata.size()-1).dup());
			z = hiddenlayer[layer].forward(z.dup());
			FloatMatrix mask = dropout(z.rows,z.columns);
			dropoutmask.add(mask);
//			System.out.println(z.rows+"Z"+z.columns);
//			System.out.println("Z"+z);
//			System.out.println(mask.rows+"mask"+mask.columns);
			inputdata.add(z.muli(mask).dup());
//			System.out.println("M"+z);
		}

		/*
		for(int layer=0; layer<hiddenlayer_N; layer++){
			float[] data; //入力値を保持
			float[][] z_ = new float[minibatchSize][hidden_N[layer]];
			int[][] mask = new int[minibatchSize][hidden_N[layer]];
			for(int n=0; n<minibatchSize; n++){
				if(layer == 0){
					//入力層ではdataにx[0]の値を代入
					data = x[n];
				}else{
					data = z[n];
				}
				z_[n] = hiddenlayer[layer].forward(data);
				mask[n] = dropout(z_[n]);
			}

			z = z_;
			inputdata.add(z.clone());
			dropoutmask.add(mask);
		}*/

		//出力
		FloatMatrix dy = logisticlayer.train(z, label, minibatchSize, l_rate);
//		float[][] dy = logisticlayer.train(z, label, minibatchSize, l_rate);
		//System.out.println("input:"+logisticlayer.input_N+" output:"+logisticlayer.output_N);
		//逆伝播
		for(int layer = hiddenlayer_N - 1; layer >= 0; layer--){
//			float[][] weight_prev;
			FloatMatrix weight_prev;
			if(layer == hiddenlayer_N-1){
				//逆伝播の一番最初(出力層)の場合
				weight_prev = logisticlayer.weight.dup();
			}else{
				weight_prev = hiddenlayer[layer+1].weight.dup();
			}
			//System.out.println(dy[minibatchSize/2].length+":"+dropoutmask.get(layer)[minibatchSize/2].length);
			dy = hiddenlayer[layer].backward(inputdata.get(layer), inputdata.get(layer+1), dy, weight_prev, minibatchSize, l_rate);

			FloatMatrix mask = dropoutmask.get(layer);
//			System.out.println(mask.isRowVector()+"/\\\\\"+mask.isColumnVector());
//			System.out.println(mask);
//			System.out.println(dy.columns+"--"+mask.rows+"**"+mask.columns);
			if(dy.columns == mask.columns){
				dy.muli(mask);
//				System.out.println(dy);
			}

//			for(int i=0; i<minibatchSize; i++){
//				FloatMatrix mask = dropoutmask.get(layer);
//				//出力にドロップアウトマスクを適用
//				//System.out.println(dy[i].length+":"+mask.length);
//				if(dy.getRow(i).columns == mask.columns){
////					if(dy[i].length == mask.length){
//					//System.out.println(dy[i].length+":"+mask.length);
//					for(int j=0; j< dy[i].length; j++){
//						dy[i][j] *= mask[j];
//					}
//				}
//			}

		}
		//System.out.println("next loop");
	}

	/**
	 * ドロップアウトマスクをかける
	 * @param row 出力の列
	 * @param column 出力の行
	 * @param columns
	 * @return マスク
	 */
	public FloatMatrix dropout( int row, int column){
//		public int[] dropout(float[] z){
//		int size = z.length;
		FloatMatrix mask = new FloatMatrix(row,column);
		//double ddd = 1.f - pdrop;
//		FloatMatrix mask = FloatMatrix.rand(z.rows, z.columns);
//		System.out.println("next:" + mask.rows +" "+mask.columns +" "+mask.length);
//		for(int i=0; i<size; i++){
//			mask[i] = RandomGenerator.binomial(1,  1.f- pdrop, mt);
//			mask.put(i, RandomGenerator.binomial(1,  1.f- pdrop, mt));
//			z[i] *= mask[i];
//		}
		for(int i=0; i<mask.rows; i++){
			for(int j=0; j<mask.columns; j++){
				mask.put(i,j,RandomGenerator.binomial(1,  1.f- pdrop, mt));
			}
		}
//			System.out.println(mask);
		//z.muli(mask);
		return mask;
	}

	/**
	 * ドロップアウトを行った層の重みをならす
	 */
	public void adjust_weight(){
		for(int layer=0; layer<hiddenlayer_N; layer++){
			int in, out;

			if(layer == 0){
				in=input_N;
			}else{
				in=hidden_N[layer-1];
				//in=hidden_N[layer];
			}

			if(layer == hiddenlayer_N-1){
				out = output_N;
				//System.out.println("layer == hiddenlayer_N -1:"+(hiddenlayer_N -1));
			}else{
				out= hidden_N[layer];
				//out = hidden_N[layer+1];
				//System.out.println(hidden_N[layer+1]);
			}
			System.out.println("layer:"+layer+" in:"+in+" out:"+out + " hiddenlength:"+hidden_N.length);
			System.out.println("hiddenlength1:"+hiddenlayer[layer].weight.length+" hiddenlength1:"+hiddenlayer[layer].weight.getRow(0).length);

			hiddenlayer[layer].weight.muli(1.f-pdrop);
//			for(int i=0; i<out; i++){
//				for(int j=0; j<in; j++){
//					hiddenlayer[layer].weight[i][j] *= (1.f-pdrop);
//				}
//			}
		}
	}


	public FloatMatrix predict(FloatMatrix x){
//	public Integer[] predict(float[] x){
		FloatMatrix z = new FloatMatrix();
//		float[] z = new float[0];

		for(int layer=0; layer<hiddenlayer_N; layer++){
//			float[] out;
			FloatMatrix out;

			if(layer == 0){
				out = x.dup();
			}else{
				out = z.dup();
			}

			z = hiddenlayer[layer].forward(out);
		}
//		z = logisticlayer.predict(z);


		return logisticlayer.predict(z);
	}

}
