package updater;

public class Updater {


	private static float param = 0.f;
	final static float omega = (float) Math.pow(1, Math.pow(10, -8));//0割をふせぐ小さい数字



	public static float SGD(float dw, float w, float learning_rate, float momentum){
		return (w - (learning_rate * dw) - (learning_rate * momentum * w));
	}


	public static float Adagrad(float dw, float w, float learning_rate, float momentum){

		float result = (w - ((learning_rate / (float)Math.sqrt(learning_rate)) * dw) - (learning_rate * momentum * w));

		return result;
	}
	public interface FloatFunction<R, S, T, U> {

		/**
		 * Applies this function to the given argument.
		 *
		 * @param value the function argument
		 * @return the function result
		 */
		R apply(float dw, float w, float learning_rate, float momentum);
	}


	public static float RMSProp(float dw, float w, float learning_rate){

		float result = (w - ((learning_rate / (float)Math.sqrt(learning_rate+omega)) * dw)); //- (learning_rate * momentum * w));
		return result;
	}


	static private int iter = 0;
	static private float mt = 0.f, vt = 0.f;

	public static float ADAM(float dw, float w, float learning_rate){
		float beta1 = 0.9f, beta2 = 0.999f;
		iter+=1;

		mt = beta1*mt + (1-beta1)*dw;
		vt = beta2*vt + (1-beta2)*dw*dw;

		float m = (float) (mt / (1 - Math.pow(beta1, iter)));
		float v = (float) (vt / (1-Math.pow(beta2, iter)));

		return (float)(w - learning_rate * m / (Math.sqrt(v)+ omega));
	}
}
