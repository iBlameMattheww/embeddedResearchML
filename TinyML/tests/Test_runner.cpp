#include "unity.h"

void setUp(void) 
{
    // Optional: code to run before each test
}

void tearDown(void) 
{
    // Optional: code to run after each test
}

/* 
activations_test.cpp content    
*/
void Test_Relu_Positive(void);
void Test_Relu_Zero(void);
void Test_Relu_Negative(void);
void Test_Exp_Approx_0(void);
void Test_Exp_Approx_Full_Range(void);
void Test_Softmax_SineDataset_SingleLogit_Returns255(void);
void test_Softmax_LorenzDataset_3Values_NormalizedAndScaled(void);
void test_Softmax_NegativeValues_StableNormalization(void);
void test_Softmax_LargeValues_NoOverflow(void);

/* 
Utils_test.cpp content    
*/
void Test_TypicalSympnetDotProduct(void);
void Test_AllZerosDotProduct(void);
void Test_MaxMagnitudeDotProduct(void);

/* 
Symplectic_tests.cpp content    
*/
void Test_SingleCoefficientPolynomialDerivation(void);
void Test_ZeroMPolynomialDerivation(void);
void Test_NegativeMPolynomialDerivation(void);
void Test_TypicalSymplecticTimeScale(void);
void Test_ZeroDerivativeSymplecticTimeScale(void);
void Test_SignHandlingSyplecticTimeScale(void);
void Test_TypicalStateUpdate(void);
void Test_ZeroScaleStateUpdate(void);
void Test_SignBehaviorStateUpdate(void);
void Test_SingleLayerSympnetStep(void);

int main(void) 
{
    UNITY_BEGIN();

    // Activations tests
    RUN_TEST(Test_Relu_Positive);
    RUN_TEST(Test_Relu_Zero);
    RUN_TEST(Test_Relu_Negative);
    RUN_TEST(Test_Exp_Approx_0);
    RUN_TEST(Test_Exp_Approx_Full_Range);
    RUN_TEST(Test_Softmax_SineDataset_SingleLogit_Returns255);
    RUN_TEST(test_Softmax_LorenzDataset_3Values_NormalizedAndScaled);
    RUN_TEST(test_Softmax_NegativeValues_StableNormalization);
    RUN_TEST(test_Softmax_LargeValues_NoOverflow);

    // Utils tests
    RUN_TEST(Test_TypicalSympnetDotProduct);
    RUN_TEST(Test_AllZerosDotProduct);
    RUN_TEST(Test_MaxMagnitudeDotProduct);

    // Symplectic tests
    RUN_TEST(Test_SingleCoefficientPolynomialDerivation);
    RUN_TEST(Test_ZeroMPolynomialDerivation);   
    RUN_TEST(Test_NegativeMPolynomialDerivation);
    RUN_TEST(Test_TypicalSymplecticTimeScale);
    RUN_TEST(Test_ZeroDerivativeSymplecticTimeScale);
    RUN_TEST(Test_SignHandlingSyplecticTimeScale);
    RUN_TEST(Test_TypicalStateUpdate);
    RUN_TEST(Test_ZeroScaleStateUpdate);
    RUN_TEST(Test_SignBehaviorStateUpdate);
    RUN_TEST(Test_SingleLayerSympnetStep);

    return UNITY_END();
}