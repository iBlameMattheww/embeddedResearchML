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
void Test_Vanilla_Init_SetsLayerCount(void);
void Test_VanillaStep_IsDeterministic(void);
void Test_VanillaStep_StepSizeScaling(void);
void Test_VanillaStep_ZeroStepSize_NoChange(void);
void Test_VanillaStep_StateRemainsFinite(void);

/* 
Utils_test.cpp content    
*/
void Test_TypicalSympnetDotProduct(void);
void Test_AllZerosDotProduct(void);

/* 
Symplectic_tests.cpp content    
*/
void Test_SingleCoefficientPolynomialDerivation(void);
void Test_ZeroMPolynomialDerivation(void);
void Test_NegativeMPolynomialDerivation(void);
void Test_TypicalSymplecticTimeScale(void);
void Test_ZeroDerivativeSymplecticTimeScale(void);
void Test_SignHandlingSymplecticTimeScale(void);
void Test_TypicalStateUpdate(void);
void Test_ZeroScaleStateUpdate(void);
void Test_SignBehaviorStateUpdate(void);
void Test_SingleLayerSympnetStep(void);
// void Test_MultiLayerSympnetStep(void);

int main(void) 
{
    UNITY_BEGIN();

    // Activations tests
    RUN_TEST(Test_Vanilla_Init_SetsLayerCount);
    RUN_TEST(Test_VanillaStep_IsDeterministic);
    RUN_TEST(Test_VanillaStep_StepSizeScaling);
    RUN_TEST(Test_VanillaStep_ZeroStepSize_NoChange);
    RUN_TEST(Test_VanillaStep_StateRemainsFinite);

    // Utils tests
    RUN_TEST(Test_TypicalSympnetDotProduct);
    RUN_TEST(Test_AllZerosDotProduct);

    // Symplectic tests
    RUN_TEST(Test_SingleCoefficientPolynomialDerivation);
    RUN_TEST(Test_ZeroMPolynomialDerivation);   
    RUN_TEST(Test_NegativeMPolynomialDerivation);
    RUN_TEST(Test_TypicalSymplecticTimeScale);
    RUN_TEST(Test_ZeroDerivativeSymplecticTimeScale);
    RUN_TEST(Test_SignHandlingSymplecticTimeScale);
    RUN_TEST(Test_TypicalStateUpdate);
    RUN_TEST(Test_ZeroScaleStateUpdate);
    RUN_TEST(Test_SignBehaviorStateUpdate);
    RUN_TEST(Test_SingleLayerSympnetStep);

    return UNITY_END();
}