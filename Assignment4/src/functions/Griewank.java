package functions;/*
 * Griewank.java
 *
 * Created on January 12, 2003, 3:50 PM
 *
 * 
 * Copyright (C) 2003 - 2006 
 * Computational Intelligence Research Group (CIRG@UP)
 * Department of Computer Science 
 * University of Pretoria
 * South Africa
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA 
 *   
 */
import java.util.*;

/**
 * Characteristics:
 * <ul>
 * <li>Multi-modal</li>
 * <li>Non seperable</li>
 * <li>Regular</li>
 * </ul>
 * 
 * f(x) = 0; x = (0,0,...,0);
 * x_i e (-600,600)
 * 
 * @author  Edwin Peer
 */
public class Griewank extends ContinuousFunction {
    
    public Griewank() 
    {
        setDimension(30);
    }
    
    public double evaluate(ArrayList<Double> x) {
        double sumsq = 0;
        double prod = 1;
        for (int i = 0; i < getDimension(); ++i) {
            sumsq += x.get(i) * x.get(i);
            prod *= Math.cos(x.get(i) / Math.sqrt(i + 1));
        }
        return 1 + sumsq * (1.0/4000.0) - prod;
    }
    
}
