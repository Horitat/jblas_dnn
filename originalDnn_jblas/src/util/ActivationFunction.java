package util;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

/**
 * 活性化関数を定義するクラス
 *
 * */
public class ActivationFunction {
	/**ステップ関数*/
	public static float step_function(float x){
		if(x > 0.f){
			return 1.f;
		}
		return 0;
	}

	/**ステップ関数*/
	public static FloatMatrix step_function(FloatMatrix x){
		return x.gt(0);
	}

	/**
	 * ロジスティックシグモイド
	 * @param x 入力
	 * @return シグモイド関数で活性化した結果
	 */
	public static float logistic_sigmoid(float x){
		return (float) (1. / (1. + Math.exp(-x)));
	}
	/**
	 *ロジスティックシグモイドの微分
	 * @param y 出力
	 * @return ロジスティックシグモイドの微分
	 */
	public static float dsigmoid(float y) {
		return (float) (y * (1. - y));
	}

	/**
	 * ロジスティックシグモイド
	 * @param x 入力
	 * @return シグモイド関数で活性化した結果
	 */
	public static FloatMatrix logistic_sigmoid(FloatMatrix x){

//		FloatMatrix y = x.dup();
		return MatrixFunctions.exp(x.neg()).add(1).rdiv(1);
	}
	/**
	 *ロジスティックシグモイドの微分
	 * @param y 出力
	 * @return ロジスティックシグモイドの微分
	 */
	public static FloatMatrix dsigmoid(FloatMatrix y) {
//		return logistic_sigmoid(y.dup()).mul(logistic_sigmoid(y.dup()).rsub(1));
		return y.mul(y.dup().rsub(1));
	}

	/**
	 * シンプルシグモイド
	 * @param x 入力
	 * @return 活性化した結果
	 */
	public static float simple_sigmoid(float x){
		return 1.f / (1.f + x);
	}
	/**
	 *シンプルシグモイドの微分
	 * @param y 出力
	 * @return シンプルシグモイドの微分
	 */
	public static float dsimple_sigmoid(float y) {
		return (float) ((-(1. + y))*(-(1. + y)));

	}

	/**ハイパーボリックタンジェント*/
	public static float tanh(float x) {
		return (float) Math.tanh(x);
	}
	/**ハイパーボリックタンジェントの微分*/
	public static float dtanh(float y) {
		return 1.f - y * y;
	}

	/**ハイパーボリックタンジェント*/
	public static FloatMatrix tanh(FloatMatrix x) {
		return MatrixFunctions.tanh(x);
	}

	/**ハイパーボリックタンジェントの微分*/
	public static FloatMatrix dtanh(FloatMatrix y) {
		//System.out.println("dtanh");
		//MatrixFunctions.pow(y, y);
		return  y.mul(y).rsub(1.f);
	}

	/**ReLU関数*/
	public static float ReLU(float x) {
		if(x > 0) {
			//System.out.println("relu");
			return x;
		} else {
			return 0.f;
		}
	}
	/**ReLUの微分*/
	public static float dReLU(float y) {
		if(y > 0) {
			//System.out.println("drelu");
			return 1.f;
		} else {
			return 0.f;
		}
	}

	/**ReLU関数*/
	public static FloatMatrix ReLU(FloatMatrix x) {
		return x.mul(x.dup().gt(0));
	}
	/**ReLUの微分*/
	public static FloatMatrix dReLU(FloatMatrix y) {
		return y.gt(0);
	}

	/**ソフトマックス関数*/
	public static float[] softmax(float[] activation, int output_N) {
		// TODO 自動生成されたメソッド・スタブ
		float[] out = new float[output_N];
		float max = 0.f;
		float sum = 0.f;
		for(float n : activation){
			if(max < n){
				max = n;
			}
			//System.out.println("act:"+n);
		}

		for(int i=0; i<output_N; i++){
			out[i] = (float) Math.exp(activation[i] - max);
			sum += out[i];
		}

		for(int i=0; i<output_N; i++){
			out[i] = out[i] / sum;

		}

		return out;
	}

	public static FloatMatrix softmax(FloatMatrix activation, int output_N) {
		FloatMatrix out = new FloatMatrix(activation.rows, output_N);

		for(int i=0; i<activation.rows; i++){
//			System.out.println(out.getRow(i));
//			float max = activation.getRow(i).max();
			out.putRow(i, MatrixFunctions.exp(activation.getRow(i).sub(activation.getRow(i).max())));
			out.putRow(i, out.getRow(i).div(out.getRow(i).sum()));
//			System.out.println(out.getRow(i));
		}
//		System.out.println(out);
//		float sum = out.sum();
//		out = MatrixFunctions.exp(activation.sub(max));
//		sum=out.sum();
		return out;
	}





	public static void main(String[] args) {
		FloatMatrix dd = new FloatMatrix(new float[][]{
			{0.0f, 50.25f, 100f, 200f, 0.45f}
		});

//		float[] aa = {0.0f, 50.25f, 100f, 200f, 0.45f};
		float[] aa = dd.toArray();
		//int[] b = {5000,1,2,3,4,5};
		//System.out.println (b);
		for(int i = 0; i< aa.length; i++){
			System.out.print(logistic_sigmoid(aa[i])+",");
		}
		System.out.print(logistic_sigmoid(dd));
		//System.out.println (aa[i]);
//		System.out.println (softmax(aa, 3));
	}

	@FunctionalInterface
	public interface FloatFunction<R> {

		/**
		 * Applies this function to the given argument.
		 *
		 * @param value the function argument
		 * @return the function result
		 */
		R apply(float value);
	}


	@FunctionalInterface
	public interface FloatMatrixFunction<R> {

		/**
		 * Applies this function to the given argument.
		 *
		 * @param value the function argument
		 * @return the function result
		 */
		R apply(FloatMatrix value);
	}



}
