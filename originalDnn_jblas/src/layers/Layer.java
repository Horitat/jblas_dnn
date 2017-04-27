package layers;

import lombok.Data;
import updater.Updater;
import util.Common_method;

@Data
public abstract class Layer {

	enum LayerType {
		CONVOLUTIONAL,POOLING,MULTILAYER,NORMALIZATION,BATCHNORM
	}


	String layername;
	float leraning_rate;
	float gradient;
	//モーメンタム、momentum*前のgrad_weightを現時点の重みの更新量に足す
	float momentum;
	//重みの減衰、l_rate*w_decay*weightを現時点の重みの更新量から引く
	float w_decay;
	Common_method.FloatFunction<Float> activation;
	Common_method.FloatFunction<Float> dactivation;
	int flatsize;
	String solver;
	//Updater updater;
	Updater.FloatFunction<Float,Float, Float, Float> updater;

	//	protected Layer(String actfunc, String updater){
	//
	//		if(actfunc.equals("sigmoid")){
	//			activation = (float x)->ActivationFunction.logistic_sigmoid(x);
	//			dactivation = (float x)->ActivationFunction.dsigmoid(x);
	//		}else if(actfunc.equals("tanh")){
	//			activation = (float x)->ActivationFunction.tanh(x);
	//			dactivation = (float x)->ActivationFunction.dtanh(x);
	//		}else if(actfunc.equals("ReLU")){
	//			activation = (float x)->ActivationFunction.ReLU(x);
	//			dactivation = (float x)->ActivationFunction.dReLU(x);
	//		}else if(StringUtils.isEmpty(actfunc)){
	//			throw new IllegalArgumentException("specify activation function");
	//		}else{
	//			throw new IllegalArgumentException("activation function not supported");
	//		}
	//
	//		if(updater.equals("sigmoid")){
	//			this.updater = (float dw, float w, float learning_rate, float momentum)->Updater.Adagrad(dw, w, learning_rate, momentum);
	//		}else if(updater.equals("tanh")){
	//			this.updater = (float dw, float w, float learning_rate, float momentum)->Updater.Adagrad(dw, w, learning_rate, momentum);
	//		}else if(updater.equals("ReLU")){
	//			this.updater = (float dw, float w, float learning_rate, float momentum)->Updater.SGD(dw, w, learning_rate, momentum);
	//		}else if(StringUtils.isEmpty(updater)){
	//			throw new IllegalArgumentException("specify updater function");
	//		}else{
	//			throw new IllegalArgumentException("updater function not supported");
	//		}
	//	}

}
