#ifndef confidence_bounds_hpp
#define confidence_bounds_hpp
#include "mdp.hpp"
#include <math.h>
#include <string>
#include <unordered_map>

double evaluateExpectedReturn(const vector<unsigned int> & policy, 
                    const MDP* evalMDP, double eps);
double evaluateExpectedReturn(vector<vector<double> > & policy, 
                    const MDP* evalMDP, double eps);                    
                    
void policyValueIteration(const vector<unsigned int> & policy, 
                    const MDP* evalMDP, double eps, double* V);
                    
void policyValueIteration(vector<vector<double> > & policy, 
                    const MDP* evalMDP, double eps, double* V);
                    
double getExpectedReturn(const MDP* mdp);
double getAverageValueFromStartStates(double* V, bool* init, unsigned int numStates);
double** calculateStateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps);
double* calculateExpectedFeatureCounts(vector<vector<double> > & policy, FeatureGridMDP* fmdp, double eps);
double* calculateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps);
double* calculateEmpiricalExpectedFeatureCounts(vector<vector<pair<unsigned int,unsigned int> > > trajectories, FeatureGridMDP* fmdp);

double evaluateExpectedReturn(const vector<unsigned int> & policy, 
                    const MDP* evalMDP, double eps)
{
    //initialize values to zero
    unsigned int numStates = evalMDP->getNumStates();
    double V[numStates];
    for(unsigned int i=0; i<numStates; i++) V[i] = 0.0;
    
    //get value of policy in evalMDP
    policyValueIteration(policy, evalMDP, eps, V);
    
    //get expected value of policy evaluated on evalMDP over starting dist
    //TODO assumes uniform distibution over start states
    bool* init = evalMDP->getInitialStates();
    return getAverageValueFromStartStates(V, init, numStates);
}

//Stochastic policy version
double evaluateExpectedReturn(vector<vector<double> > & policy, 
                    const MDP* evalMDP, double eps)
{
    //initialize values to zero
    unsigned int numStates = evalMDP->getNumStates();
    double V[numStates];
    for(unsigned int i=0; i<numStates; i++) V[i] = 0.0;
    
    //get value of policy in evalMDP
    policyValueIteration(policy, evalMDP, eps, V);
    
    //get expected value of policy evaluated on evalMDP over starting dist
    //TODO assumes uniform distibution over start states
    bool* init = evalMDP->getInitialStates();
    return getAverageValueFromStartStates(V, init, numStates);
}


vector<double> evaluateExpectedReturnVector(const vector<unsigned int> & policy, 
                    const MDP* evalMDP, double eps)
{
    //initialize values to zero
    vector<double> init_state_returns;
    unsigned int numStates = evalMDP->getNumStates();
    double V[numStates];
    for(unsigned int i=0; i<numStates; i++) V[i] = 0.0;
    
    //get value of policy in evalMDP
    policyValueIteration(policy, evalMDP, eps, V);
    
    //get expected value of policy evaluated on evalMDP over starting dist
    //TODO assumes uniform distibution over start states
    bool* init = evalMDP->getInitialStates();
    //check if there is at least one starting state
    bool startStateExists = false;
    for(unsigned int i=0; i<numStates; i++)
        if(init[i])
            startStateExists = true;
    assert(startStateExists);

    for(unsigned int s=0; s < numStates; s++)
    {
        if(init[s])
        {
            init_state_returns.push_back(V[s]);
        }
    }
    return init_state_returns;

}




double getAverageValueFromStartStates(double* V, bool* init, unsigned int numStates)
{
    //check if there is at least one starting state
    bool startStateExists = false;
    for(unsigned int i=0; i<numStates; i++)
        if(init[i])
            startStateExists = true;
    assert(startStateExists);
    double valSum = 0;
    int initCount = 0;
    for(unsigned int s=0; s < numStates; s++)
    {
        if(init[s])
        {
            valSum += V[s];
            initCount++;
        }
    }
    return valSum / initCount;
}

//Updates vector of values V to be value of using policy in evalMDP
//run value iteration until convergence using policy actions rather than argmax
void policyValueIteration(const vector<unsigned int> & policy, 
                    const MDP* evalMDP, double eps, double* V)
{
    double delta;
    double discount = evalMDP->getDiscount();
    double*** T = evalMDP->getTransitions();
    //repeat until convergence within error eps
    do
    {
        unsigned int numStates = evalMDP->getNumStates();

        //cout << "--------" << endl;
        //displayAsGrid(V);
        delta = 0;
        //update value of each state
       // cout << eps * (1 - discount) / discount << "," << delta << endl;
        
        for(unsigned int s1 = 0; s1 < numStates; s1++)
        {
            double tempV = 0;
            //add reward
            tempV += evalMDP->getReward(s1);
            //add discounted value of next state based on policy action
            int policy_action = policy[s1];
            //calculate expected utility of taking action a in state s1
            double expUtil = 0;
            
            for(unsigned int s2 = 0; s2 < numStates; s2++)
            {
                expUtil += T[s1][policy_action][s2] * V[s2];
            }
            tempV += discount * expUtil;

            //update delta to track convergence
            double absDiff = abs(tempV - V[s1]);
            if(absDiff > delta)
                delta = absDiff;
            V[s1] = tempV;
        }
        
    }
    while(delta > eps);

}

//Stochastic policy version
//Updates vector of values V to be value of using policy in evalMDP
//run value iteration until convergence using policy actions rather than argmax
void policyValueIteration(vector<vector<double> > & policy, 
                    const MDP* evalMDP, double eps, double* V)
{
    double delta;
    double discount = evalMDP->getDiscount();
    double*** T = evalMDP->getTransitions();
    //repeat until convergence within error eps
    do
    {
        unsigned int numStates = evalMDP->getNumStates();
        unsigned int numActions = evalMDP->getNumActions();

        //cout << "--------" << endl;
        //displayAsGrid(V);
        delta = 0;
        //update value of each state
       // cout << eps * (1 - discount) / discount << "," << delta << endl;
        
        for(unsigned int s1 = 0; s1 < numStates; s1++)
        {
            double tempV = 0;
            //add reward
            tempV += evalMDP->getReward(s1);
            //add discounted value of next state based on stochastic policy action
            for(unsigned int a = 0; a < numActions; a++)
            {
                double policy_action_prob = policy[s1][a];
                //calculate expected utility of taking action a in state s1
                double expUtil = 0;
                
                for(unsigned int s2 = 0; s2 < numStates; s2++)
                {
                    expUtil += T[s1][a][s2] * V[s2];
                }
                tempV += discount * policy_action_prob * expUtil;
            
            }

            //update delta to track convergence
            double absDiff = abs(tempV - V[s1]);
            if(absDiff > delta)
                delta = absDiff;
            V[s1] = tempV;
        }
        
    }
    while(delta > eps);

}

//returns the expected return of the optimal policy for the input mdp
//assumes value iteration has already been run
double getExpectedReturn(const MDP* mdp)
{
    unsigned int numStates = mdp->getNumStates();
    double* V = mdp->getValues();
    bool* init = mdp->getInitialStates();
    return getAverageValueFromStartStates(V, init, numStates);

}

//returns the expected return of the optimal policy for the input mdp
//assumes value iteration has already been run
vector<double> getExpectedReturnVector(const MDP* mdp)
{
    vector<double> init_state_returns;
    unsigned int numStates = mdp->getNumStates();
    double* V = mdp->getValues();
    bool* init = mdp->getInitialStates();

    //check if there is at least one starting state
    bool startStateExists = false;
    for(unsigned int i=0; i<numStates; i++)
        if(init[i])
            startStateExists = true;
    assert(startStateExists);

    for(unsigned int s=0; s < numStates; s++)
    {
        if(init[s])
        {
            init_state_returns.push_back(V[s]);
        }
    }
    return init_state_returns;
}

//uses an analogue to policy evaluation to calculate the expected features for each state
//runs until change is less than eps
double** calculateStateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps)
{
    unsigned int numStates = fmdp->getNumStates();
    unsigned int numFeatures = fmdp->getNumFeatures();
    double** stateFeatures = fmdp->getStateFeatures();
    double discount = fmdp->getDiscount();
    double*** T = fmdp->getTransitions();


    //initalize 2-d array for storing feature weights
    double** featureCounts = new double*[numStates];
    for(unsigned int s = 0; s < numStates; s++)
        featureCounts[s] = new double[numFeatures];
    for(unsigned int s = 0; s < numStates; s++)
        for(unsigned int f = 0; f < numFeatures; f++)
            featureCounts[s][f] = 0;

    //run feature count iteration
    double delta;
    
    //repeat until convergence within error eps
    do
    {
            //////Debug
//           for(unsigned int s = 0; s < numStates; s++)
//            {
//                double* fcount = featureCounts[s];
//                cout << "State " << s << ": ";
//                for(unsigned int f = 0; f < numFeatures; f++)
//                    cout << fcount[f] << "\t";
//                cout << endl;
//            }    
//            cout << "-----------" << endl;
            //////////

        //cout << "--------" << endl;
        //displayAsGrid(V);
        delta = 0;
        //update value of each state
       // cout << eps * (1 - discount) / discount << "," << delta << endl;
        
        for(unsigned int s1 = 0; s1 < numStates; s1++)
        {
            //cout << "for state: " << s1 << endl;
            //use temp array to store accumulated, discounted feature counts
            double tempF[numFeatures];
            for(unsigned int f = 0; f < numFeatures; f++)
                tempF[f] = 0;            
            
            //add current state features
            for(unsigned int f =0; f < numFeatures; f++)
                tempF[f] += stateFeatures[s1][f];

            //update value of each reachable next state following policy
            unsigned int policyAction = policy[s1];
            double transitionFeatures[numFeatures];
            for(unsigned int f = 0; f < numFeatures; f++)
                transitionFeatures[f] = 0;

            for(unsigned int s2 = 0; s2 < numStates; s2++)
            {
                if(T[s1][policyAction][s2] > 0)
                {       
                    //cout << "adding transition to state: " << s2 << endl;
                    //accumulate features for state s2
                    for(unsigned int f = 0; f < numFeatures; f++)
                        transitionFeatures[f] += T[s1][policyAction][s2] * featureCounts[s2][f];
                }
            }
            //add discounted transition features to tempF
            for(unsigned int f = 0; f < numFeatures; f++)
            {
                tempF[f] += discount * transitionFeatures[f];
                //update delta to track convergence
                double absDiff = abs(tempF[f] - featureCounts[s1][f]);
                if(absDiff > delta)
                    delta = absDiff;
                featureCounts[s1][f] = tempF[f];
            }
        }
        //cout << "delta " << delta << endl;
    }
    while(delta > eps);

    return  featureCounts;
}


//uses an analogue to policy evaluation to calculate the expected features for each state
//runs until change is less than eps
///Stochastic policy version!!!
double** calculateStateExpectedFeatureCounts(vector<vector<double> > & policy, FeatureGridMDP* fmdp, double eps)
{
    unsigned int numStates = fmdp->getNumStates();
    unsigned int numActions = fmdp->getNumActions();
    unsigned int numFeatures = fmdp->getNumFeatures();
    double** stateFeatures = fmdp->getStateFeatures();
    double discount = fmdp->getDiscount();
    double*** T = fmdp->getTransitions();


    //initalize 2-d array for storing feature weights
    double** featureCounts = new double*[numStates];
    for(unsigned int s = 0; s < numStates; s++)
        featureCounts[s] = new double[numFeatures];
    for(unsigned int s = 0; s < numStates; s++)
        for(unsigned int f = 0; f < numFeatures; f++)
            featureCounts[s][f] = 0;

    //run feature count iteration
    double delta;
    
    //repeat until convergence within error eps
    do
    {
            //////Debug
//           for(unsigned int s = 0; s < numStates; s++)
//            {
//                double* fcount = featureCounts[s];
//                cout << "State " << s << ": ";
//                for(unsigned int f = 0; f < numFeatures; f++)
//                    cout << fcount[f] << "\t";
//                cout << endl;
//            }    
//            cout << "-----------" << endl;
            //////////

        //cout << "--------" << endl;
        //displayAsGrid(V);
        delta = 0;
        //update value of each state
       // cout << eps * (1 - discount) / discount << "," << delta << endl;
        
        for(unsigned int s1 = 0; s1 < numStates; s1++)
        {
            //cout << "for state: " << s1 << endl;
            //use temp array to store accumulated, discounted feature counts
            double tempF[numFeatures];
            for(unsigned int f = 0; f < numFeatures; f++)
                tempF[f] = 0;            
            
            //add current state features
            for(unsigned int f =0; f < numFeatures; f++)
                tempF[f] += stateFeatures[s1][f];

            //update value of each reachable next state following policy
            double transitionFeatures[numFeatures];
            for(unsigned int f = 0; f < numFeatures; f++)
                transitionFeatures[f] = 0;

            for(unsigned int s2 = 0; s2 < numStates; s2++)
            {
                for(unsigned int a = 0; a < numActions; a++)
                {
                    if(T[s1][a][s2] > 0 && policy[s1][a] > 0)
                    {       
                        //cout << "adding transition to state: " << s2 << endl;
                        //accumulate features for state s2
                        for(unsigned int f = 0; f < numFeatures; f++)
                            transitionFeatures[f] += policy[s1][a] * T[s1][a][s2] * featureCounts[s2][f];
                    }
                }
            }
            //add discounted transition features to tempF
            for(unsigned int f = 0; f < numFeatures; f++)
            {
                tempF[f] += discount * transitionFeatures[f];
                //update delta to track convergence
                double absDiff = abs(tempF[f] - featureCounts[s1][f]);
                if(absDiff > delta)
                    delta = absDiff;
                featureCounts[s1][f] = tempF[f];
            }
        }
        //cout << "delta " << delta << endl;
    }
    while(delta > eps);

    return  featureCounts;
}


//Stochastic policy version. 
double* calculateExpectedFeatureCounts(vector<vector<double> > & policy, FeatureGridMDP* fmdp, double eps)
{
    //average over initial state distribution (assumes all initial states equally likely)
    double** stateFcounts = calculateStateExpectedFeatureCounts(policy, fmdp, eps);
    unsigned int numStates = fmdp -> getNumStates();
    unsigned int numFeatures = fmdp -> getNumFeatures();
    int numInitialStates = 0;
    
    double* expFeatureCounts = new double[numFeatures];
    fill(expFeatureCounts, expFeatureCounts + numFeatures, 0);
    
    for(unsigned int s = 0; s < numStates; s++)
        if(fmdp -> isInitialState(s))
        {
            numInitialStates++;
            for(unsigned int f = 0; f < numFeatures; f++)
                expFeatureCounts[f] += stateFcounts[s][f];
        }
                
    //divide by number of initial states
    for(unsigned int f = 0; f < numFeatures; f++)
        expFeatureCounts[f] /= numInitialStates;
    
    //clean up
    for(unsigned int s = 0; s < numStates; s++)
        delete[] stateFcounts[s];
    delete[] stateFcounts;    
    
    return expFeatureCounts;

}


double* calculateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps)
{
    //average over initial state distribution (assumes all initial states equally likely)
    double** stateFcounts = calculateStateExpectedFeatureCounts(policy, fmdp, eps);
    unsigned int numStates = fmdp -> getNumStates();
    unsigned int numFeatures = fmdp -> getNumFeatures();
    int numInitialStates = 0;
    
    double* expFeatureCounts = new double[numFeatures];
    fill(expFeatureCounts, expFeatureCounts + numFeatures, 0);
    
    for(unsigned int s = 0; s < numStates; s++)
        if(fmdp -> isInitialState(s))
        {
            numInitialStates++;
            for(unsigned int f = 0; f < numFeatures; f++)
                expFeatureCounts[f] += stateFcounts[s][f];
        }
                
    //divide by number of initial states
    for(unsigned int f = 0; f < numFeatures; f++)
        expFeatureCounts[f] /= numInitialStates;
    
    //clean up
    for(unsigned int s = 0; s < numStates; s++)
        delete[] stateFcounts[s];
    delete[] stateFcounts;    
    
    return expFeatureCounts;

}

double* calculateEmpiricalExpectedFeatureCounts(vector<vector<pair<unsigned int,unsigned int> > > trajectories, FeatureGridMDP* fmdp)
{
    unsigned int numFeatures = fmdp->getNumFeatures();
    double gamma = fmdp->getDiscount();
    double** stateFeatures = fmdp->getStateFeatures();

    //average over all trajectories the discounted feature weights
    double* aveFeatureCounts = new double[numFeatures];
    fill(aveFeatureCounts, aveFeatureCounts + numFeatures, 0);
    for(vector<pair<unsigned int, unsigned int> > traj : trajectories)
    {
        for(unsigned int t = 0; t < traj.size(); t++)
        {
            pair<unsigned int, unsigned int> sa = traj[t];
            unsigned int state = sa.first;
            //cout << "adding features for state " << state << endl;
            //for(unsigned int f = 0; f < numFeatures; f++)
            //    cout << stateFeatures[state][f] << "\t";
            //cout << endl;
            for(unsigned int f = 0; f < numFeatures; f++)
                aveFeatureCounts[f] += pow(gamma, t) * stateFeatures[state][f];
        }
    }
    //divide by number of demos
    for(unsigned int f = 0; f < numFeatures; f++)
        aveFeatureCounts[f] /= trajectories.size();
    return aveFeatureCounts;
}




//calculate based on demos and policy and take infintity norm of difference
double calculateWorstCaseFeatureCountBound(vector<unsigned int> & policy, FeatureGridMDP* fmdp, vector<vector<pair<unsigned int,unsigned int> > > trajectories, double eps)
{
    unsigned int numFeatures = fmdp -> getNumFeatures();
    double* muhat_star = calculateEmpiricalExpectedFeatureCounts(trajectories,
                                                                  fmdp);    
    double* mu_pieval = calculateExpectedFeatureCounts(policy, fmdp, eps);
    //calculate the infinity norm of the difference
    double maxAbsDiff = 0;
    for(unsigned int f = 0; f < numFeatures; f++)
    {
        double absDiff = abs(muhat_star[f] - mu_pieval[f]);
        if(absDiff > maxAbsDiff)
            maxAbsDiff = absDiff;
    }
    //clean up
    delete[] muhat_star;
    delete[] mu_pieval;     
    return maxAbsDiff;
}


double getPolicyLoss(FeatureGridMDP* sampleMDP, vector<unsigned int> & eval_pi, double precision)
{

    double Vstar = getExpectedReturn(sampleMDP);
    //cout << "True Exp Val" << endl;
    //cout << Vstar << endl;
    //cout << "Eval Policy" << endl; 
    double Vhat = evaluateExpectedReturn(eval_pi, sampleMDP, precision);
    //cout << Vhat << endl;
    return Vstar - Vhat;

}


/*
           BAse Cvar Implementation not yet working

def solve_cvar_expret_fixed_policy(mdp_env, u_policy, u_expert, posterior_rewards, p_R, alpha, debug=False):
    '''
    Solves for CVaR and expectation with respect to BROIL baseline regret formulation using u_expert as the baseline
    input 
        mdp_env: the mdp
        u_policy: the pre-optimized policy
        u_expert: the state-action occupancies of the expert
        posterior_rewards: a matrix with each column a reward hypothesis or each column a weight vector
        p_R: a posterior probability mass function over the reward hypotheses
        alpha: the risk sensitivity, higher is more conservative. We look at the (1-alpha)*100% average worst-case
        lamda: the amount to weight expected return versus CVaR if 0 then fully robust, if 1, then fully return maximizing
        returns: tuple (u_cvar, cvar) the occupancy frequencies of the policy optimal wrt cvar and the actual cvar value
    '''


    num_states, num_actions, gamma = mdp_env.num_states, mdp_env.num_actions, mdp_env.gamma
    weight_dim, n = posterior_rewards.shape  #weight_dim is dimension of reward function weights and n is the number of samples in the posterior
    #get number of state-action occupancies

    #NOTE: k may be much larger than weight_dim! This isn't the k dim of the weights!
    k = mdp_env.num_states * mdp_env.num_actions

    #need to redefine R if using linear reward approximation with features and feature weights since R will be weight vectors
    R = np.zeros((k,n))
    for i in range(n):
        #print(posterior_rewards[:,i])
        R[:,i] = mdp_env.transform_to_R_sa(posterior_rewards[:,i]) #this method is overwritten by each MDP class to do the right thing
        #print(np.reshape(R[:25,i],(5,5)))
    #print(R)
    #input()

    posterior_probs = p_R
    #new objective is 
    #max \sigma - 1/(1-\alpha) * p^T z for vector of auxiliary variables z.

    #so the decision variables are (in the following order) sigma, and all the z's from the ReLUs

    #we want to maximize so take the negative of this vector and minimize via scipy 
    c_cvar = -1. * np.concatenate((np.ones(1),                 #for sigma
                        -1.0/(1.0 - alpha) * posterior_probs))  #for the auxiliary variables z

    #constraints: for each of the auxiliary variables we have a constraint >=0 and >= the stuff inside the ReLU

    #create constraint for each auxiliary variable should have |R| + 1 (for sigma) + n (for samples) columns 
    # and n rows (one for each z variable)
    auxiliary_constraints = np.concatenate((np.ones((n,1)), -np.eye(n)),axis=1)
    
    #add the upper bounds for these constraints:
    #check to see if we have mu or u
    if k != len(u_expert):
        #we have feature approximation and have mu rather than u, at least we should
        #print(weight_dim)
        #print(u_expert)
        assert len(u_expert) == weight_dim
        print(R.shape)
        print(u_policy.shape)
        print(posterior_rewards.transpose().shape)
        print(u_expert.shape)

        auxiliary_b = np.dot(R.transpose(), u_policy) - np.dot(posterior_rewards.transpose(), u_expert)
    else:
        #no feature approximation for reward, just tabular
        auxiliary_b = np.dot(R.transpose(), u_policy - u_expert)

    #add the non-negativitity constraints for z(R). 
    auxiliary_z_geq0 = np.concatenate((np.zeros((n,1)), -np.eye(n)), axis=1)
    auxiliary_bz_geq0 = np.zeros(n)

    
    A_cvar = np.concatenate((auxiliary_constraints,
                            auxiliary_z_geq0), axis=0)
    b_cvar = np.concatenate((auxiliary_b, auxiliary_bz_geq0))

    #solve the LP
    sol = linprog(c_cvar, A_ub=A_cvar, b_ub = b_cvar, bounds=(None, None)) #TODO:might be good to explicitly make the bounds here rather than via constraints...
    if debug: print("solution to optimizing CVaR")
    if debug: print(sol)
    
    if sol['success'] is False:
        #print(sol)
        print("didn't solve correctly!")
        input("Continue?")
    #the solution of the LP corresponds to the CVaR
    var_sigma = sol['x'][0] #get sigma (this is VaR (at least close))
    cvar = -sol['fun'] #get the cvar of the input policy (negative since we minimized negative objective)
    
    #calculate expected return of optimized policy
    if k != len(u_expert):
        #we have feature approximation and have mu rather than u, at least we should
        #print(weight_dim)
        #print(u_expert)
        assert len(u_expert) == weight_dim
        expected_perf_expert = np.dot(posterior_probs, np.dot(posterior_rewards.transpose(), u_expert))
    else:
        expected_perf_expert = np.dot( np.dot(R, posterior_probs), u_expert)
    cvar_exp_ret = np.dot( np.dot(R, posterior_probs), u_policy) - expected_perf_expert

    if debug: 
        print("CVaR = ", cvar)
        print("Expected return = ", cvar_exp_ret)
    
    
    return cvar, cvar_exp_ret



def solve_max_cvar_policy(mdp_env, u_expert, posterior_rewards, p_R, alpha, debug=False, lamda = 0.0):
    '''input mdp_env: the mdp
        u_expert: the state-action occupancies of the expert
        posterior_rewards: a matrix with each column a reward hypothesis or each column a weight vector
        p_R: a posterior probability mass function over the reward hypotheses
        alpha: the risk sensitivity, higher is more conservative. We look at the (1-alpha)*100% average worst-case
        lamda: the amount to weight expected return versus CVaR if 0 then fully robust, if 1, then fully return maximizing
        returns: tuple (u_cvar, cvar) the occupancy frequencies of the policy optimal wrt cvar and the actual cvar value
    '''


    num_states, num_actions, gamma = mdp_env.num_states, mdp_env.num_actions, mdp_env.gamma
    weight_dim, n = posterior_rewards.shape  #weight_dim is dimension of reward function weights and n is the number of samples in the posterior
    #get number of state-action occupancies

    #NOTE: k may be much larger than weight_dim!
    k = mdp_env.num_states * mdp_env.num_actions

    #need to redefine R if using linear reward approximation with features and feature weights since R will be weight vectors
    R = np.zeros((k,n))
    for i in range(n):
        #print(posterior_rewards[:,i])
        R[:,i] = mdp_env.transform_to_R_sa(posterior_rewards[:,i]) #this method is overwritten by each MDP class to do the right thing
        #print(np.reshape(R[:25,i],(5,5)))
    #print(R)
    #input()

    posterior_probs = p_R
    #new objective is 
    #max \sigma - 1/(1-\alpha) * p^T z for vector of auxiliary variables z.

    #so the decision variables are (in the following order) all the u(s,a) and sigma, and all the z's.

    #we want to maximize so take the negative of this vector and minimize via scipy 
    u_coeff = np.dot(R, posterior_probs)
    c_cvar = -1. * np.concatenate((lamda * u_coeff, #for the u(s,a)'s (if lamda = 0 then no in objective, this is the lambda * p^T R^T u)
                        (1-lamda) * np.ones(1),                 #for sigma
                        (1-lamda) * -1.0/(1.0 - alpha) * posterior_probs))  #for the auxiliary variables z

    #constraints: for each of the auxiliary variables we have a constraint >=0 and >= the stuff inside the ReLU

    #create constraint for each auxiliary variable should have |R| + 1 (for sigma) + n (for samples) columns 
    # and n rows (one for each z variable)
    auxiliary_constraints = np.zeros((n, k + 1 + n))
    for i in range(n):
        z_part = np.zeros(n)
        z_part[i] = -1.0 #make the part for the auxiliary variable >= the part in the relu
        z_row = np.concatenate((-R[:,i],  #-R_i(s,a)'s
                                np.ones(1),    #sigma
                                z_part))
        auxiliary_constraints[i,:] = z_row

    #add the upper bounds for these constraints:
    #check to see if we have mu or u
    if k != len(u_expert):
        #we have feature approximation and have mu rather than u, at least we should
        #print(weight_dim)
        #print(u_expert)
        assert len(u_expert) == weight_dim
        auxiliary_b = -1. * np.dot(posterior_rewards.transpose(), u_expert)
    else:
        auxiliary_b = -1. * np.dot(R.transpose(), u_expert)

    #add the non-negativitity constraints for the vars u(s,a) and z(R). 
    #mu's greater than or equal to zero
    auxiliary_u_geq0 = -np.eye(k, M=k+1+n)  #negative since constraint needs to be Ax<=b
    auxiliary_bu_geq0 = np.zeros(k)

    auxiliary_z_geq0 = np.concatenate((np.zeros((n, k+1)), -np.eye(n)), axis=1)
    auxiliary_bz_geq0 = np.zeros(n)

    #don't forget the normal MDP constraints over the mu(s,a) terms
    I_s = np.eye(num_states)
    I_minus_gamma_Ps = []
    for P_a in mdp_env.get_transition_prob_matrices():
        I_minus_gamma_Ps.append(I_s - gamma * P_a.transpose())

    A_eq = np.concatenate(I_minus_gamma_Ps, axis=1)

    # if mdp_env.num_actions == 4:
    #     A_eq = np.concatenate(( I_s - gamma * mdp_env.P_left.transpose(),
    #                             I_s - gamma * mdp_env.P_right.transpose(),
    #                             I_s - gamma * mdp_env.P_up.transpose(),
    #                             I_s - gamma * mdp_env.P_down.transpose()),axis =1)
    # else:
    #     A_eq = np.concatenate((I_s - gamma * mdp_env.P_left.transpose(),
    #                            I_s - gamma * mdp_env.P_right.transpose()),axis =1)
    b_eq = mdp_env.init_dist
    A_eq_plus = np.concatenate((A_eq, np.zeros((mdp_env.num_states,1+n))), axis=1)  #add zeros for sigma and the auxiliary z's

    A_cvar = np.concatenate((auxiliary_constraints,
                            auxiliary_u_geq0,
                            auxiliary_z_geq0), axis=0)
    b_cvar = np.concatenate((auxiliary_b, auxiliary_bu_geq0, auxiliary_bz_geq0))

    #solve the LP
    sol = linprog(c_cvar, A_eq=A_eq_plus, b_eq = b_eq, A_ub=A_cvar, b_ub = b_cvar, bounds=(None, None)) #TODO:might be good to explicitly make the bounds here rather than via constraints...
    if debug: print("solution to optimizing CVaR")
    if debug: print(sol)
    
    if sol['success'] is False:
        #print(sol)
        print("didn't solve correctly!")
        input("Continue?")
    #the solution of the LP corresponds to the CVaR
    var_sigma = sol['x'][k] #get sigma (this is VaR (at least close))
    cvar_opt_usa = sol['x'][:k]

    #calculate the CVaR of the solution
    if k != len(u_expert):
        relu_part = var_sigma * np.ones(n) - np.dot(np.transpose(R), cvar_opt_usa) + np.dot(np.transpose(posterior_rewards), u_expert)
    else:
        relu_part = var_sigma * np.ones(n) - np.dot(np.transpose(R), cvar_opt_usa) + np.dot(np.transpose(R), u_expert)
    #take max with zero
    relu_part[relu_part < 0] = 0.0
    cvar = var_sigma - 1.0/(1 - alpha) * np.dot(posterior_probs, relu_part)

    #calculate expected return of optimized policy
    if k != len(u_expert):
        #we have feature approximation and have mu rather than u, at least we should
        #print(weight_dim)
        #print(u_expert)
        assert len(u_expert) == weight_dim
        exp_baseline_perf = np.dot(posterior_probs, np.dot(posterior_rewards.transpose(), u_expert))
    
    else:
        exp_baseline_perf = np.dot(np.dot(R, posterior_probs), u_expert)


    cvar_exp_ret = np.dot( np.dot(R, posterior_probs), cvar_opt_usa) - exp_baseline_perf

    if debug: print("CVaR = ", cvar)
    if debug: print("policy u(s,a) = ", cvar_opt_usa)
    cvar_opt_stoch_pi = utils.get_optimal_policy_from_usa(cvar_opt_usa, mdp_env)
    if debug: print("CVaR opt stochastic policy")
    if debug: print(cvar_opt_stoch_pi)

    if debug:
        if k != len(u_expert):
            policy_losses = np.dot(R.transpose(), cvar_opt_usa)  - np.dot(posterior_rewards.transpose(), u_expert)
        else:
            policy_losses = np.dot(R.transpose(), cvar_opt_usa - u_expert)
        print("policy losses:", policy_losses)
    if debug: 
        if k != len(u_expert):
            print("expert returns:", np.dot(posterior_rewards.transpose(), u_expert))
        else:
            print("expert returns:", np.dot(R.transpose(), u_expert))
    if debug: print("my returns:", np.dot(R.transpose(), cvar_opt_usa))

    return cvar_opt_usa, cvar, cvar_exp_ret


def solve_minCVaR_reward(mdp_env, u_expert, posterior_rewards, p_R, alpha):
    '''
    Solves the dual problem
      input:
        mdp_env: the mdp
        u_expert: the state-action occupancies of the expert
        R: a matrix with each column a reward hypothesis
        p_R: a posterior probability mass function over the reward hypotheses
        alpha: the risk sensitivity, higher is more conservative. We look at the (1-alpha)*100% average worst-case
       output:
        The adversarial reward and the q weights on the reward posterior. Optimizing for this reward should yield the CVaR optimal policy
    '''
    weight_dim, n = posterior_rewards.shape  #weight_dim is dimension of reward function weights and n is the number of samples in the posterior
    #get number of state-action occupancies

    #NOTE: k may be much larger than weight_dim!
    k = mdp_env.num_states * mdp_env.num_actions

    num_states, num_actions, gamma = mdp_env.num_states, mdp_env.num_actions, mdp_env.gamma
    p0 = mdp_env.init_dist

    #need to redefine R if using linear reward approximation with features and feature weights since R will be weight vectors
    R = np.zeros((k,n))
    for i in range(n):
        #print(posterior_rewards[:,i])
        R[:,i] = mdp_env.transform_to_R_sa(posterior_rewards[:,i]) #this method is overwritten by each MDP class to do the right thing
        #print(np.reshape(R[:25,i],(5,5)))
    #print(R)
    #input()


    #k,n = R.shape  #k is dimension of reward function and n is the number of samples in the posterior
    posterior_probs = p_R
    #objective is min p_0^Tv - u_E^T R q

    #the decision variables are (in the following order) q (an element for each reward in the posterior) and v(s) for all s
    #coefficients on objective
    if k != len(u_expert):
        #function approximation, u_expert is expected feature counts
        c_q = np.concatenate((np.dot(-posterior_rewards.transpose(), u_expert), p0))  #for the auxiliary variables z
    else:
        c_q = np.concatenate((np.dot(-R.transpose(), u_expert), p0))  #for the auxiliary variables z

    #constraints: 

    #sum of q's should equal 1
    A_eq = np.concatenate((np.ones((1,n)), np.zeros((1,num_states))), axis = 1)
    b_eq = np.ones(1)
    
    #leq constraints

    #first do the q <= 1/(1-alpha) p
    A_q_leq_p = np.concatenate((np.eye(n), np.zeros((n, num_states))), axis=1)
    b_q_leq_p = 1.0/(1 - alpha) * p_R

    #next do the value iteration equations
    I_s = np.eye(num_states)
    #TODO: debug this and check it is more general using Ps (see cvar method)
    I_minus_gamma_Ps = []
    for P_a in mdp_env.get_transition_prob_matrices():
        I_minus_gamma_Ps.append(I_s - gamma * P_a)

    trans_dyn = np.concatenate(I_minus_gamma_Ps, axis=0)

    # if mdp_env.num_actions == 4:
    #     trans_dyn = np.concatenate(( I_s - gamma * mdp_env.P_left,
    #                             I_s - gamma * mdp_env.P_right,
    #                             I_s - gamma * mdp_env.P_up,
    #                             I_s - gamma * mdp_env.P_down), axis=0)
    # else:
    #     trans_dyn = np.concatenate((I_s - gamma * mdp_env.P_left,
    #                            I_s - gamma * mdp_env.P_right), axis=0)
    
    A_vi = np.concatenate((R, -trans_dyn), axis=1)
    b_vi = np.zeros(num_states * num_actions)

    #last add constraint that all q >= 0
    A_q_geq_0 = np.concatenate((-np.eye(n), np.zeros((n, num_states))), axis=1)
    b_q_geq_0 = np.zeros(n)

    #stick them all together
    A_leq = np.concatenate((A_q_leq_p,
                            A_vi,
                            A_q_geq_0), axis=0)
    b_geq = np.concatenate((b_q_leq_p, b_vi, b_q_geq_0))

    #solve the LP
    sol = linprog(c_q, A_eq=A_eq, b_eq=b_eq, A_ub=A_leq, b_ub=b_geq, bounds=(None, None)) #TODO:might be good to explicitly make the bounds here rather than via constraints...
    #print("solution to optimizing for CVaR reward")
    #print(sol)
    cvar = sol['fun'] #I think the objective value should be the same?
    #the solution of the LP corresponds to the CVaR
    q = sol['x'][:n] #get sigma (this is VaR (at least close))
    values = sol['x'][n:]
    print("CVaR = ", cvar)
    print("policy v(s) under Rq = ", values)
    print("expected value", np.dot(mdp_env.init_dist, values))
    
    #print("q weights:", q)
    cvar_reward_fn = np.dot(R,q)
    #print("CVaR reward Rq =", cvar_reward_fn)

    return cvar_reward_fn, q

*/

#endif
