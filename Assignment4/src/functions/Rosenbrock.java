package functions;/*
 * Rosenbrock.java
 *
 * Created on January 12, 2003, 2:18 PM
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
 *
 * Characteristics:
 * <ul>
 * <li>Unimodal</li>
 * <li>Non Separable</li>
 * <li>Continuous</li>
 * </ul>
 *
 * f(x) = 0; x = 1
 *
 * x e [-2.048,2.048]
 *
 * @author  Edwin Peer
 */
public class Rosenbrock extends ContinuousFunction {
    
    public Rosenbrock() {
        setDimension(30);
    }
    
    
    public double evaluate(ArrayList<Double> x) 
    {
        double tmp = 0;
        for (int i = 0; i < getDimension()-1; ++i) {
            double a = x.get(i);
            double b = x.get(i + 1);
       
            tmp += ((100 * (b-a*a) * (b-a*a)) + ((a-1.0) * (a-1.0)));
        }

        return tmp;
    }
    
}
