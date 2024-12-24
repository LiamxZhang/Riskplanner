#
import torch
from configs import CONTROL_PARAMS, RISK_PARAMS

class FatalityRisk:
    """
    Paper: UAV path optimization with an integrated cost assessment model considering third-party risks in metropolitan environments
    """
    def __init__(self):
        # The crash probability of UAV system
        
        self.uas_position = torch.tensor([0.0,0.0,0.0],dtype=torch.float32)
        self.uas_velocity = torch.tensor([0.0,0.0,0.0],dtype=torch.float32)
        self.m = torch.tensor(CONTROL_PARAMS["mass"], dtype=torch.float32)
        self.g = torch.tensor(CONTROL_PARAMS["gravity"], dtype=torch.float32)
        self.S_hit = torch.tensor(RISK_PARAMS["impact_area"], dtype=torch.float32)
        self.R = torch.tensor(RISK_PARAMS["drag_coefficient"], dtype=torch.float32)
        self.rho_A = torch.tensor(RISK_PARAMS["air_density"], dtype=torch.float32) # air density kg/m^3

    def calculate_risk_cost(self):
        return


    
    def update_uas_state(self, pos, vel):
        self.uas_position = torch.tensor(pos, dtype=torch.float32)
        self.uas_velocity = torch.tensor(vel, dtype=torch.float32)

    def update_pedestrian_state(self, pos, vel):
        self.ped_position = torch.tensor(pos, dtype=torch.float32)
        self.ped_velocity = torch.tensor(vel, dtype=torch.float32)

    def calculate_cr_p(self):
        self.P_crash = 0.1 
        # Number of hitted pedestrian
        N_hit_p = self.calculate_hit_pedestrian()
        # Casualty
        R_f_p = self.calculate_casualty()
        Cr_p =  self.P_crash * N_hit_p * R_f_p
        return Cr_p

    def calculate_hit_pedestrian(self):
        sigma_p = self.calculate_population_density()
        return self.S_hit * sigma_p

    def calculate_population_density(self):
        # Falling point on the ground
        fall_time = self.calculate_falling_time()

        # Nearby pedestrians

        return
    
    def calculate_casualty(self):
        alpha = torch.tensor(106, dtype=torch.float32)
        beta = torch.tensor(100, dtype=torch.float32)
        E_imp = self.calculate_kinetic_energy()
        S_C = torch.tensor(0.5, dtype=torch.float32)
        return 1 / (1 + torch.sqrt(alpha / beta) * ( (beta / E_imp) ** (1 / (4 * S_C)) ) )

    def calculate_kinetic_energy(self):
        """
        Calculate the kinetic energy when falling on the ground
        """
        velocity_copy = self.uas_velocity.clone()  
        velocity_copy[2] = self.calculate_falling_velocity()
        velocity_norm_squared = torch.dot(velocity_copy, velocity_copy)  # Equivalent to ||v||^2
        kinetic_energy = 0.5 * self.m * velocity_norm_squared
        return kinetic_energy

    def calculate_falling_time(self):
        """
        Calculate the falling time when falling on the ground
        """
        self._epsilon = torch.sqrt((2 * self.g) / self._xi)

        term1 = torch.exp(-self.h * self._xi)  #  exp(-h_e * xi)
        sqrt_term = torch.sqrt(1 - term1 ) #  sqrt(1 - exp(-h_e * xi))
        inner_log = (2 / (1 - sqrt_term)) - 1  
        t_e = (1 / (self._epsilon *  self._xi)) * torch.log(inner_log)
        return t_e

    def calculate_falling_time2(self):
        """
        Calculate the falling time when falling on the ground
        """
        self._epsilon = torch.sqrt((2 * self.g) / self._xi)
        inner_log = 2 * self._epsilon / (self._epsilon - self.calculate_falling_velocity()) - 1  
        t_e = (1 / (self._epsilon *  self._xi)) * torch.log(inner_log) 
        return t_e

    def calculate_falling_velocity(self):
        """
        Calculate the falling velocity
        """
        # Caculate height difference
        self.h = self.uas_position[2] - self.ped_position[2]
        self._xi = self.R * self.S_hit * self.rho_A / self.m
        return torch.sqrt((2 * self.g / self._xi) * (1 - torch.exp(-self.h * self._xi)))
    

if __name__ == "__main__":
    risk = FatalityRisk()

    # Test case 1: Update UAV state and pedestrian state
    risk.update_uas_state([0.0, 0.0, 10.0], [0.0, 0.0, 0.0])  # UAV at 10 meters height
    risk.update_pedestrian_state([0.0, 0.0, 1.5], [0.0, 0.0, 0.0])  # Pedestrian at 1.5 meters height

    kinetic_energy = risk.calculate_kinetic_energy()
    print(f"kinetic_energy: {kinetic_energy} J")
    time = risk.calculate_falling_time()
    print(f"falling_time: {time} seconds")
    time2 = risk.calculate_falling_time2()
    print(f"falling_time2: {time2} seconds")
    casualty = risk.calculate_casualty()
    print(f"casualty: {casualty} ")
