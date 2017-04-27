package single_layer;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.jblas.FloatMatrix;

import util.ActivationFunction;
import util.Common_method;
import util.GaussianDistribution;
import Mersenne.Sfmt;

public class Logistic_kaiki {
	public FloatMatrix weight;
	public FloatMatrix bias;
//	public float[][] weight;
//	public float[] bias;
	public int input_N;
	public int output_N;
	static BufferedReader buffer = new BufferedReader(new InputStreamReader(System.in));
	public Logistic_kaiki(int input, int output){
		input_N = input;
		output_N = output;

//		weight = new float[output_N][input_N];
//		bias = new float[output_N];
		weight = new FloatMatrix(output_N,input_N);
//		System.out.println(output_N +"   "+input);
//		System.out.println(weight);
		bias = new FloatMatrix(1,output_N);
	}

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ

		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		Sfmt mt = new Sfmt(init_key);

		int input_N = 2;//入力の数
		final int classes = 3;
		final int output_N = classes;
		final int train_N = 500 * classes;//学習データの数
		final int test_N = 100 * classes;//テストデータの数

		final int epochs = 2000;//トレーニングの最大世代数
		float l_rate = 1.0f;//学習率

		int minibatchsize = 50; //ミニバッチサイズ
		int minibatch_N = train_N / minibatchsize;//ミニバッチの数

		FloatMatrix traindata = new FloatMatrix(train_N,input_N);
		FloatMatrix trainlabel = new FloatMatrix(train_N,classes);
//		float[][] traindata = new float[train_N][input_N];
//		int[][] trainlabel = new int[train_N][classes];

		FloatMatrix testdata = new FloatMatrix(test_N,input_N);
		FloatMatrix testlabel = new FloatMatrix(test_N,classes);

		FloatMatrix predict = new FloatMatrix(test_N,classes);
//		float[][] testdata = new float[test_N][input_N];
//		Integer[][] testlabel = new Integer[test_N][classes];
//
//		Integer[][] predict = new Integer[test_N][classes];

		GaussianDistribution g1 = new GaussianDistribution(-2., 1., mt);
		GaussianDistribution g2 = new GaussianDistribution(2., 1., mt);
		GaussianDistribution g3 = new GaussianDistribution(0.0, 1., mt);

		//データの生成
		for(int i=0; i<train_N; i++){
			if(i < train_N / classes){
				traindata.put(i,0, (float) g1.random());
				traindata.put(i,1, (float) g2.random());
				trainlabel.putRow(i, new FloatMatrix(1,3,1,0,0));
//				traindata[i][0] = (float)g1.random();
//				traindata[i][1] = (float)g2.random();
//				trainlabel[i] = new int[]{1,0,0};
			}else if(train_N / classes <= i && i < (train_N / classes) * 2){
				traindata.put(i,0, (float) g2.random());
				traindata.put(i,1, (float) g1.random());
				trainlabel.putRow(i, new FloatMatrix(1,3,0,1,0));
//				traindata[i][0] = (float)g2.random();
//				traindata[i][1] = (float)g1.random();
//				trainlabel[i] = new int[]{0,1,0};
			}else{
				traindata.put(i,0, (float) g3.random());
				traindata.put(i,1, (float) g3.random());
				trainlabel.putRow(i, new FloatMatrix(1,3,0,0,1));
//				traindata[i][0] = (float)g3.random();
//				traindata[i][1] = (float)g3.random();
//				trainlabel[i] = new int[]{0,0,1};
			}
		}
		for(int i=0; i<test_N; i++){
			if(i < test_N / classes){
				testdata.put(i,0,(float)g1.random());
				testdata.put(i,1,(float)g2.random());
				testlabel.putRow(i, new FloatMatrix(1,3,1,0,0));
//				testdata[i][0] = (float)g1.random();
//				testdata[i][1] = (float)g2.random();
//				testlabel[i] = new Integer[]{1,0,0};
			}else if(test_N / classes <= i && i < (test_N / classes) * 2){
				testdata.put(i,0,(float)g2.random());
				testdata.put(i,1,(float)g1.random());
				testlabel.putRow(i, new FloatMatrix(1,3,0,1,0));
//				testdata[i][0] = (float)g2.random();
//				testdata[i][1] = (float)g1.random();
//				testlabel[i] = new Integer[]{0,1,0};
			}else{
				testdata.put(i,0,(float)g3.random());
				testdata.put(i,1,(float)g3.random());
				testlabel.putRow(i, new FloatMatrix(1,3,0,0,1));
//				testdata[i][0] = (float)g3.random();
//				testdata[i][1] = (float)g3.random();
//				testlabel[i] = new Integer[]{0,0,1};
			}
		}

		FloatMatrix[] train_minibatch = new FloatMatrix[minibatch_N];//学習の入力データ
//		float[][][] train_minibatch = new float[minibatch_N][minibatchsize][input_N];//学習の入力データ
		FloatMatrix[] train_minibatch_label = new FloatMatrix[minibatch_N];//学習のラベル
//		int[][][] train_minibatch_label = new int[minibatch_N][minibatchsize][input_N];//学習のラベル

		List<Integer> minibatchindex = new ArrayList<>(); //SGDを適用する順番
		//学習データをシャッフルするための番号
		for(int i=0; i<train_N; i++){
			minibatchindex.add(i);
		}
		//System.out.println(minibatchindex);
		Common_method.list_shuffle(minibatchindex, mt);
		//ミニバッチに分割
		//System.out.println(trainlabel.getRow(2));
		for(int i=0; i< minibatch_N; i++){
			train_minibatch[i] = new FloatMatrix(minibatchsize,input_N);
			train_minibatch_label[i] = new FloatMatrix(minibatchsize,output_N);
			for(int j=0; j<minibatchsize; j++){
				train_minibatch[i].putRow(j, traindata.getRow(minibatchindex.get(i*minibatchsize+j)));
				train_minibatch_label[i].putRow(j, trainlabel.getRow(minibatchindex.get(i*minibatchsize+j)));
//				train_minibatch[i][j] = traindata[minibatchindex.get(i*minibatchsize+j)];
//				train_minibatch_label[i][j] = trainlabel[minibatchindex.get(i*minibatchsize+j)];
//				System.out.println(traindata.getRow(minibatchindex.get(i*minibatchsize+j)));
//				System.out.println(trainlabel.getRow(minibatchindex.get(i*minibatchsize+j)));
			}
//			System.out.println(traindata.getRow(minibatchindex.get(i*minibatchsize+j)));
//			System.out.println(train_minibatch_label[i]);
		}
		Logistic_kaiki classifier = new Logistic_kaiki(input_N, output_N);
//		System.out.println(train_minibatch[0].rows +"+++++"+train_minibatch[0].columns);
		//学習実行
		for(int epoch=0; epoch<epochs; epoch++){
			for(int batch=0; batch<minibatch_N; batch++){
				classifier.train(train_minibatch[batch], train_minibatch_label[batch], minibatchsize, l_rate);
			}
			l_rate = l_rate * 0.9f;
//			System.out.println(epoch+"/"+epochs);
		}
		//テスト
		predict = classifier.predict(testdata);
//		for(int i=0; i<test_N; i++){
//			predict[i] = classifier.predict(testdata[i]);
//		}
		print_result_test(predict,testlabel, test_N, classes);

	}


	public static void print_result_test(FloatMatrix predict, FloatMatrix testlabel, int test_N, int classes){
//		public static void print_result_test(Integer[][] predict, Integer[][] testlabel, int test_N, int classes){
		FloatMatrix confusion = new FloatMatrix(classes,classes);
		float accuracy =0.f;
		FloatMatrix precision = new FloatMatrix(classes);
		FloatMatrix recall = new FloatMatrix(classes);
//		int[][] confusion = new int[classes][classes];
//		float accuracy =0.f;
//		float[] precision = new float[classes];
//		float[] recall = new float[classes];

//		System.out.println(test_N +":"+ predict.rows+","+predict.columns);
//		int[][] predict_a = predict.toIntArray2();
//		int[][] testlabel_a = testlabel.toIntArray2();
//		Matrix uj_predict = new JBlasDenseDoubleMatrix2D(predict);
		//モデルからの答えと正解を突き合わせ
		for(int i=0; i<test_N; i++){
			Integer[] predict_array = new Integer[predict.getRow(i).toIntArray().length];
			Integer[] label_array = new Integer[testlabel.getRow(i).toIntArray().length];
//			System.out.println(predict.getRow(i).toIntArray().length+"/////"+testlabel.getRow(i).toIntArray().length);
			for(int j=0; j<predict_array.length; j++){
				predict_array[j] = predict.getRow(i).toIntArray()[j];
				label_array[j] =testlabel.getRow(i).toIntArray()[j];
//				System.out.println(j);
			}


			int row = Arrays.asList(predict_array).indexOf(1);
			int column = Arrays.asList(label_array).indexOf(1);
//			int row = Arrays.asList(predict.getRow(i).toIntArray()).indexOf(1);
//			int column = Arrays.asList(testlabel.getRow(i).toIntArray()).indexOf(1);
//			System.out.println(predict.getRow(i));
//			System.out.println(Arrays.asList(predict.getRow(i).toArray()[i]));
//			System.out.println("######################################");
//			System.out.println(testlabel.getRow(i));
//			System.out.println("lllllllllllllllllllllllllllllllllllllllllll\n");

//			if(row != -1 && column != -1){
				confusion.put(row,column, confusion.get(row,column)+1f);
//			}
//			int row = Arrays.asList(predict[i]).indexOf(1);
//			int column = Arrays.asList(testlabel[i]).indexOf(1);
//			System.out.println(i +":"+ row+","+column);
//			confusion[row][column]++;
		}
		System.out.println(confusion);
		for(int i=0; i<classes; i++){
			accuracy += confusion.get(i,i);
			//System.out.println("column "+ i + ":"+ confusion.getColumn(i).sum());
			precision.put(i, (precision.get(i)+confusion.get(i,i))/confusion.getColumn(i).sum());
//			System.out.println("row "+ i + ":"+ confusion.getRow(i).sum());
			recall.put(i, (recall.get(i)+confusion.get(i,i))/confusion.getRow(i).sum());
		}
//		for(int i=0; i<classes; i++){
//			int col = 0, row = 0;
//			for(int j=0; j<classes; j++){
//				if(i==j){
//					accuracy += confusion[i][j];
//					precision[i] += confusion[j][i];
//					recall[i] += confusion[i][j];
//				}
//				col += confusion[j][i];
//				row += confusion[i][j];
//			}
//			precision[i] = precision[i]/(float)col;
//			recall[i] = recall[i]/(float)row;
//		}

		accuracy = accuracy / test_N;

		System.out.println("------------------------------------");
		System.out.println("Logistic Regression model evaluation");
		System.out.println("------------------------------------");
		System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
		System.out.println("Precision:");
		//System.out.println(precision);
		for (int i = 0; i < classes; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, precision.get(i) * 100);
		}
		System.out.println("Recall:");
		//System.out.println(recall);
		for (int i = 0; i < classes; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, recall.get(i) * 100);
		}

	}

	/**予測メソッド*/
	public FloatMatrix predict(FloatMatrix data) {
//		public Integer[] predict(float[] data) {
		// TODO 自動生成されたメソッド・スタブ
//		System.out.println(data.rows + "###"+ data.columns);
		FloatMatrix result = output(data);
//		Integer[] label = new Integer[output_N];
//		result = result.transpose().dup();
		FloatMatrix label = FloatMatrix.zeros(result.rows, result.columns);
//		int index_max = -1;
//		float max = 0.f;
		/*for(int i=0; i<output_N; i++){
			if(max < result[i]){
				max = result[i];
				index_max = i;
			}
			//System.out.println("result:"+ result[i]);
		}
		for(int i=0; i<output_N; i++){
			if(i == index_max){
				label[i] = 1;
			}else{
				label[i] = 0;
			}
			//System.out.println("label:"+ label[i]);
		}*/

//		System.out.println("############");
//		System.out.println(result);
//		System.out.println("#####ss#######");
//		System.out.println(data);
		for(int i=0; i<result.rows; i++){
//			System.out.println(i+"   "+result.getRow(i).argmax());
			label.put(i, result.getRow(i).argmax(), 1.f);
		}
		//System.out.println(label);
//		System.out.println(result.argmax());
//		System.out.println(result.rows+"############"+result.columns);
		return label;
//		return label.put(result.argmax(), 1);
	}

	/**予測メソッド*/
	/*public Integer[] predict(double[] data) {
		// TODO 自動生成されたメソッド・スタブ

		double[] result = output(data);
		Integer[] label = new Integer[output_N];
		int index_max = -1;
		double max = 0.;

		for(int i=0; i<output_N; i++){
			if(max < result[i]){
				max = result[i];
				index_max = i;
			}
			//System.out.println("result:"+ result[i]);
		}
		for(int i=0; i<output_N; i++){
			if(i == index_max){
				label[i] = 1;
			}else{
				label[i] = 0;
			}
			//System.out.println("label:"+ label[i]);
		}
		//System.out.println("predict end");
		return label;
	}*/

	/**
	 * 出力層のトレーニングメッソド。結果に基づきパラメータ更新
	 * Softmax-crossentropy
	 * @param data 入力データ
	 * @param label 正解ラベル
	 * @param minibatchsize ミニバッチサイズ
	 * @param l_rate 学習率
	 * @return 正解ラベルとの差異
	 */
	public FloatMatrix train(FloatMatrix data, FloatMatrix label, int minibatchsize, float l_rate) {
//		public float[][] train(float[][] data, int[][] label, int minibatchsize, float l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		FloatMatrix grad_weight = new FloatMatrix(output_N,input_N);
		FloatMatrix grad_bias = new FloatMatrix(1,output_N);
		FloatMatrix error = new FloatMatrix(minibatchsize,output_N);
//		float[][] grad_weight = new float[output_N][input_N];
//		float[] grad_bias = new float[output_N];
//		float[][] error = new float[minibatchsize][output_N];
		FloatMatrix result = output(data, minibatchsize);

//		System.out.println(result.rows +"&%&&"+result.columns);
//		System.out.println("-----------------");
//		System.out.println(label.rows+"))))))))))))"+ label.columns);
		//System.out.println(result);
//		System.out.println("-------");

		error = result.sub(label).dup();
		//System.out.println(error);
//		try {
//			buffer.readLine();
//		} catch (IOException e) {
//			// TODO 自動生成された catch ブロック
//			e.printStackTrace();
//		}
//		System.out.println(error.columnSums().transpose() +"&%&&"+error.columnSums().transpose().columns);
//		System.out.println("-----------------");
//		System.out.println(data.rows+"))))))))))))"+ data.columns);
//		System.out.println("-------");
		grad_weight = error.transpose().mmul(data).dup();
		grad_bias = error.columnSums().transpose().dup();
//		System.out.println(grad_weight);
		grad_weight.muli(l_rate);
//		System.out.println("##############");
//		System.out.println(grad_weight);
//		try {
//			buffer.readLine();
//		} catch (IOException e) {
//			// TODO 自動生成された catch ブロック
//			e.printStackTrace();
//		}
		weight.subi(grad_weight.div((float)minibatchsize));
		bias.subi(grad_bias.mul(l_rate).div((float)minibatchsize));
//		weight.subi(grad_weight.mul(l_rate).div((float)minibatchsize));
//		bias.subi(grad_bias.mul(l_rate).div((float)minibatchsize));

		/*for(int i=0; i<minibatchsize; i++){
			//出力層の数だけ出力を計算
			float[] result = output(data[i]);

			for(int j=0; j<output_N; j++){
				//それぞれの出力の誤差を計算
				error[i][j] = result[j] - label[i][j];

				for(int n=0; n<input_N; n++){
					//勾配を計算
					grad_weight[j][n] += error[i][j] * data[i][n];
				}
				//バイアスの勾配を計算
				grad_bias[j] += error[i][j];
			}
		}
		//パラメータの更新
		for(int i=0; i<output_N; i++){
			for(int j=0; j<input_N; j++){
				//weight[i][j] = weight[i][j] - l_rate * grad_weight[i][j] / minibatchsize;
				weight[i][j] -= l_rate * grad_weight[i][j] / minibatchsize;
			}
//			bias[i] = bias[i] - l_rate * grad_bias[i] / minibatchsize;
			bias[i] -= l_rate * grad_bias[i] / minibatchsize;
		}*/
		return error;
	}

	/**
	 * 出力層のトレーニングメッソド。結果に基づきパラメータ更新
	 * @param data 入力データ
	 * @param label 正解ラベル
	 * @param minibatchsize ミニバッチサイズ
	 * @param l_rate 学習率
	 * @return 正解ラベルとの差異
	 */
	/*public double[][] train(double[][] data, int[][] label, int minibatchsize, double l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		double[][] grad_weight = new double[output_N][input_N];
		double[] grad_bias = new double[output_N];
		double[][] error = new double[minibatchsize][output_N];

		for(int i=0; i<minibatchsize; i++){
			//出力層の数だけ出力を計算
			double[] result = output(data[i]);

			for(int j=0; j<output_N; j++){
				//それぞれの出力の誤差を計算
				error[i][j] = result[j] - label[i][j];

				for(int n=0; n<input_N; n++){
					//勾配を計算
					grad_weight[j][n] += error[i][j] * data[i][n];
				}
				//バイアスの勾配を計算
				grad_bias[j] += error[i][j];
			}
		}
		//パラメータの更新
		for(int i=0; i<output_N; i++){
			for(int j=0; j<input_N; j++){
				weight[i][j] = (float) (weight[i][j] - l_rate * grad_weight[i][j] / minibatchsize);
			}
			bias[i] = (float) (bias[i] - l_rate * grad_bias[i] / minibatchsize);
		}
		return error;
	}*/

	/**
	 * 出力を計算
	 * @param input_data 入力データ
	 * @return 計算結果
	 */
	private FloatMatrix output(FloatMatrix input_data, int minibatch) {
//		private float[] output(float[] input_data) {
		FloatMatrix activation = new FloatMatrix(minibatch,output_N);
//		float[] activation = new float[output_N];

//		System.out.println(minibatch+"   "+weight.rows+"$$"+ weight.columns);
//		System.out.println(minibatch+"   "+input_data.rows+"^^^^"+ input_data.columns);
		/*if(input_data.isRowVector()){
			activation = weight.mmul(input_data.transpose()).dup();
		}else{
			activation = weight.mmul(input_data).dup();
		}*/
		activation = input_data.mmul(weight.transpose()).dup();
		activation.addiRowVector(bias);
//		System.out.println(bias);
//		System.out.println(bias.rows+"lll"+bias.columns);
		/*
		for(int i=0; i< output_N; i++){
			for(int j=0; j< input_N; j++){
				//activation[i] = activation[i] + (weight[i][j] * input_data[j]);
				activation[i] += weight[i][j] * input_data[j];
				//System.out.println("weight:"+weight[i][j]);
			}
			activation[i] += bias[i];
			//System.out.println("act:"+activation[i]);
		}*/

//		System.out.println(minibatch+"   "+ActivationFunction.softmax(activation, output_N).rows+"^^^^"+ ActivationFunction.softmax(activation, output_N).columns);
//		System.out.println("weight:"+weight);
//		System.out.println("softmax:"+ActivationFunction.softmax(activation, output_N));
//		System.out.println(activation);
//		try {
//			buffer.readLine();
//		} catch (IOException e) {
//			// TODO 自動生成された catch ブロック
//			e.printStackTrace();
//		}
		return ActivationFunction.softmax(activation, output_N);
	}

	/**
	 * 出力を計算
	 * @param input_data 入力データ
	 * @return 計算結果
	 */
	private FloatMatrix output(FloatMatrix input_data) {
//		private float[] output(float[] input_data) {
		FloatMatrix activation = new FloatMatrix(input_data.rows,output_N);
//		float[] activation = new float[output_N];

//		System.out.println(input_data.rows+"act:"+input_data.columns);
//		System.out.println(input_data);
//		if(input_data.isRowVector()){
//			activation = weight.mmul(input_data.transpose()).dup();
//		}else{
//			activation = input_data.mmul(weight.transpose()).dup();
//			activation = weight.mmul(input_data.transpose()).dup();
//		}
		activation = input_data.mmul(weight.transpose()).dup();
		activation.addiRowVector(bias);
		/*
		for(int i=0; i< output_N; i++){
			for(int j=0; j< input_N; j++){
				//activation[i] = activation[i] + (weight[i][j] * input_data[j]);
				activation[i] += weight[i][j] * input_data[j];
				//System.out.println("weight:"+weight[i][j]);
			}
			activation[i] += bias[i];
			//System.out.println("act:"+activation[i]);
		}*/
//		System.out.println("::"+activation.rows+"^^^^"+ activation.columns);
//		System.out.println("::"+input_data.rows+"    "+ input_data.columns);
//		activation = activation.transpose().dup();
//		System.out.println("   "+ActivationFunction.softmax(activation, output_N).rows+"^^^^"+ ActivationFunction.softmax(activation, output_N).columns);
//		System.out.println("   "+ActivationFunction.softmax(activation, output_N));

		return ActivationFunction.softmax(activation, output_N);
	}


	/**
	 * 出力を計算
	 * @param input_data 入力データ
	 * @return 計算結果
	 */
	/*private double[] output(double[] input_data) {
		double[] activation = new double[output_N];

		for(int i=0; i< output_N; i++){
			for(int j=0; j< input_N; j++){
				activation[i] = activation[i] + (weight[i][j] * input_data[j]);
				//System.out.println("weight:"+weight[i][j]);
			}
			activation[i] += bias[i];
			//System.out.println("act:"+activation[i]);
		}
		return ActivationFunction.softmax(activation, output_N);
	}*/
}
