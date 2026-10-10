import java.math.BigInteger;

public class PrimeChecker {

    public static boolean isPrime(BigInteger n) {
        // certainty = higher → lower chance of false positive
        int certainty = 20; 
        return n.isProbablePrime(certainty);
    }

}