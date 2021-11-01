package functions;

import java.util.*;

public abstract class ContinuousFunction
{
	private int Dimension = 1;
    
    public void setDimension(int d)
    {
    	Dimension = d;
    }
    
    public int getDimension()
    {
    	return Dimension;
    }
    
    
    public double getMinimum() 
    {
        return 0;
    }
    
    public abstract double evaluate(ArrayList<Double> x); 
	
}