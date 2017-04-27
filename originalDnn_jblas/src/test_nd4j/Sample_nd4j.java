package test_nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Sample_nd4j {

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		INDArray x = Nd4j.create(new double[][] {
				{1, 2},
				{3, 4}
		});

		INDArray y = Nd4j.create(new double[][] {
				{5, 6},
				{7, 8}
		});
		// (a)
		System.out.println( x.add(y) );

		System.out.println("-----");
		// (b)
		System.out.println( x.mmul(y) );

		System.out.println("-----");
		// (c)
		System.out.println( x.transpose() );
	}

}
