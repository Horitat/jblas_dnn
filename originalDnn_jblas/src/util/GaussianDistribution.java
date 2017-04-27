package util;

import Mersenne.Sfmt;



public class GaussianDistribution {
	private final double mean;
	private final double var;
	private final Sfmt mt;

	public GaussianDistribution(double mean, double var, Sfmt mt) {
		if (var < 0.0) {
			throw new IllegalArgumentException("Variance must be non-negative value.");
		}
		//Sfmt mt = new Sfmt((int)(Runtime.getRuntime().freeMemory()/(1024)));
		this.mean = mean;
		this.var = var;

		if (mt == null) {
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			//Sfmt mt = new Sfmt((int)(Runtime.getRuntime().freeMemory()/(1024)));
			mt = new Sfmt(init_key);
		}
		this.mt = mt;
	}

	public double random() {
		double r = 0.0;
		while (r == 0.0) {
			r = mt.NextUnif();
		}

		double c = Math.sqrt( -2.0 * Math.log(r) );

		if (mt.NextUnif() < 0.5) {
			return c * Math.sin( 2.0 * Math.PI * mt.NextUnif() ) * var + mean;
		}

		return c * Math.cos( 2.0 * Math.PI * mt.NextUnif() ) * var + mean;
	}
}
