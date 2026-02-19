"""
GR4J Hydrological Model Implementation
4-parameter daily rainfall-runoff model
"""

import numpy as np

class GR4J:
    """
    GR4J Model (Génie Rural à 4 paramètres Journalier)
    
    Parameters:
    - X1: Production store capacity (mm)
    - X2: Groundwater exchange coefficient (mm)
    - X3: Routing store capacity (mm)  
    - X4: Unit hydrograph time base (days)
    """
    
    def __init__(self, X1=350, X2=0, X3=90, X4=1.7):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        
    def run(self, precip, evap):
        """
        Run GR4J model
        
        Args:
            precip: Precipitation (mm/day)
            evap: Potential evapotranspiration (mm/day)
            
        Returns:
            Simulated streamflow (mm/day)
        """
        
        n = len(precip)
        
        # Initialize states
        S = self.X1 * 0.5  # Production store
        R = self.X3 * 0.5  # Routing store
        
        # Output array
        Q = np.zeros(n)
        
        # Unit hydrographs
        UH1, UH2 = self._compute_unit_hydrographs()
        
        # Routing stores
        UH1_stores = np.zeros(len(UH1))
        UH2_stores = np.zeros(len(UH2))
        
        for t in range(n):
            # Net precipitation/evaporation
            if precip[t] >= evap[t]:
                Pn = precip[t] - evap[t]
                En = 0
                
                # Production store
                capacity_ratio = S / self.X1
                Ps = self.X1 * (1 - capacity_ratio**2) * np.tanh(Pn / self.X1)
                Ps = Ps / (1 + capacity_ratio * np.tanh(Pn / self.X1))
                
                Es = 0
            else:
                Pn = 0
                En = evap[t] - precip[t]
                
                # Evaporation from store
                capacity_ratio = S / self.X1
                Es = S * (2 - capacity_ratio) * np.tanh(En / self.X1)
                Es = Es / (1 + (1 - capacity_ratio) * np.tanh(En / self.X1))
                
                Ps = 0
            
            # Update production store
            S = S - Es + Ps
            S = np.clip(S, 0, self.X1)
            
            # Percolation
            perc_ratio = S / self.X1
            perc = S * (1 - (1 + (perc_ratio / 2.25)**4)**(-0.25))
            S = S - perc
            
            # Routing
            Pr = perc + (precip[t] - Ps)
            Pr9 = 0.9 * Pr
            Pr1 = 0.1 * Pr
            
            # Unit hydrograph routing
            UH1_stores = np.roll(UH1_stores, 1)
            UH1_stores[0] = Pr9
            Q9 = np.sum(UH1_stores * UH1)
            
            UH2_stores = np.roll(UH2_stores, 1)
            UH2_stores[0] = Pr1
            Q1 = np.sum(UH2_stores * UH2)
            
            # Groundwater exchange
            F = self.X2 * (R / self.X3)**3.5
            
            # Update routing store
            R = max(0, R + Q9 + F)
            
            # Outflow
            routing_ratio = R / self.X3
            Qr = R * (1 - (1 + (routing_ratio / 2.25)**4)**(-0.25))
            R = R - Qr
            
            # Total flow
            Qd = max(0, Qr + Q1)
            Q[t] = Qd
        
        return Q
    
    def _compute_unit_hydrographs(self):
        """Compute UH1 and UH2 ordinates"""
        
        # UH1
        nUH1 = int(np.ceil(self.X4))
        UH1 = np.zeros(nUH1)
        
        for t in range(nUH1):
            if t < self.X4:
                UH1[t] = ((t + 1) / self.X4)**2.5
        
        if nUH1 > 1:
            UH1[1:] = UH1[1:] - UH1[:-1]
        
        # UH2
        nUH2 = int(np.ceil(2 * self.X4))
        UH2 = np.zeros(nUH2)
        
        for t in range(nUH2):
            if t < self.X4:
                UH2[t] = 0.5 * ((t + 1) / self.X4)**2.5
            elif t < 2 * self.X4:
                ratio = 2 - (t + 1) / self.X4
                if ratio > 0:
                    UH2[t] = 1 - 0.5 * ratio**2.5
                else:
                    UH2[t] = 1
        
        if nUH2 > 1:
            UH2[1:] = UH2[1:] - UH2[:-1]
        
        return UH1, UH2


def calculate_nse(observed, simulated):
    """Nash-Sutcliffe Efficiency"""
    obs_mean = np.mean(observed)
    numerator = np.sum((observed - simulated)**2)
    denominator = np.sum((observed - obs_mean)**2)
    nse = 1 - (numerator / denominator)
    return nse


def calculate_rmse(observed, simulated):
    """Root Mean Square Error"""
    return np.sqrt(np.mean((observed - simulated)**2))


def calculate_bias(observed, simulated):
    """Mean Bias"""
    return np.mean(simulated - observed)