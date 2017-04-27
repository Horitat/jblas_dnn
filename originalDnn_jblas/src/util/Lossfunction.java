package util;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

public class Lossfunction {

	public interface FloatFunction<R, S> {

		/**
		 * Applies this function to the given argument.
		 *
		 * @param value the function argument
		 * @return the function result
		 */
		R apply(float value, float label);
	}

	public interface FloatMatrixFunction<R, S> {

		/**
		 * Applies this function to the given argument.
		 *
		 * @param value the function argument
		 * @return the function result
		 */
		R apply(FloatMatrix value, FloatMatrix label);
	}


	public static class MSE{
		public static float mse(float data, float label){
			return (label - data)*(label - data)/2.f;
		}

		public static float dmse(float data, float label){
			return data-label;
		}

		public static FloatMatrix mse(FloatMatrix data, FloatMatrix label){
			return MatrixFunctions.pow(data.sub(label),2).rdiv(2.f);
		}

		public static FloatMatrix dmse(FloatMatrix data, FloatMatrix label){
			return data.sub(label);
		}

	}

	public static class Cross_Entropy{

		/**
		 * クロスエントロピー、多クラス分類
		 * @param data
		 * @param label
		 * @return
		 */
		public static float cross_entropy_multi(float data, float label){
			return (float) (-label * Math.log(data));
		}

		public static float dcross_entropy_multi(float data, float label){
			return (- label)/ data;
		}

		/**
		 * クロスエントロピー、２クラス分類
		 * @param data
		 * @param label
		 * @return
		 */
		public static float cross_entropy(float data, float label){
			return (-1.f)*(float)(label * Math.log(data)+(1-label)*Math.log(1-data));
		}

		public static float dcross_entropy(float data, float label){
			return (data - label)/ (data * (1-data));
		}

		/**
		 * クロスエントロピー、多クラス分類
		 * @param data
		 * @param label
		 * @return
		 */
		public static FloatMatrix cross_entropy_multi(FloatMatrix data, FloatMatrix label){
			return label.mul(MatrixFunctions.log(data)).neg();
		}

		public static FloatMatrix dcross_entropy_multi(FloatMatrix data, FloatMatrix label){
			return label.div(data).neg();
		}

		/**
		 * クロスエントロピー、２クラス分類
		 * @param data
		 * @param label
		 * @return
		 */
		public static FloatMatrix cross_entropy(FloatMatrix data, FloatMatrix label){
			return label.mul(MatrixFunctions.log(data)).add(label.rsub(1).mul(MatrixFunctions.log(data.rsub(1)))).neg();
		}

		public static FloatMatrix dcross_entropy(FloatMatrix data, FloatMatrix label){
			return data.sub(label).div(data.mul(data.rsub(1)));
		}
	}
	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ

	}

}
