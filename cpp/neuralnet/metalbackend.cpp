#ifdef USE_METAL_BACKEND

#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/metalbackend.h"
#include "../core/test.h"

namespace {
constexpr int kMetalFallbackPolicyVersion = 14;
constexpr int kMetalFallbackInputMetaChannels = 1;
thread_local vector<unique_ptr<vector<float>>> gMetalTempFloatBuffers;

static void clearMetalTempFloatBuffers() {
  gMetalTempFloatBuffers.clear();
}

static float* pinMetalTempFloatBuffer(vector<float>&& values) {
  auto buffer = std::make_unique<vector<float>>(std::move(values));
  float* ptr = buffer->data();
  gMetalTempFloatBuffers.push_back(std::move(buffer));
  return ptr;
}

static vector<float> computeMergedScale(const BatchNormLayerDesc* desc) {
  vector<float> merged(desc->numChannels);
  for(int i = 0; i < desc->numChannels; i++) {
    const float denom = std::sqrt(desc->variance[i] + desc->epsilon);
    merged[i] = desc->scale[i] / denom;
  }
  return merged;
}

static vector<float> computeMergedBias(const BatchNormLayerDesc* desc, const vector<float>& mergedScale) {
  vector<float> merged(desc->numChannels);
  for(int i = 0; i < desc->numChannels; i++) {
    merged[i] = desc->bias[i] - desc->mean[i] * mergedScale[i];
  }
  return merged;
}

static MatBiasLayerDesc makeZeroMatBiasLayerDesc(int numChannels) {
  MatBiasLayerDesc desc;
  desc.name = "metal-fallback-bias";
  desc.numChannels = numChannels;
  desc.weights.assign((size_t)numChannels, 0.0f);
  return desc;
}

}

/// Converts a ConvLayerDesc instance from C++ to Swift by creating a new SWConvLayerDesc instance with the same properties.
/// - Parameter desc: The ConvLayerDesc instance to convert.
/// - Returns: A SWConvLayerDesc instance with the same properties as the input ConvLayerDesc.
SWConvLayerDesc MetalProcess::convLayerDescToSwift(const ConvLayerDesc * desc) {

  SWConvLayerDesc swDesc = createSWConvLayerDesc(desc->convYSize,
                                                 desc->convXSize,
                                                 desc->inChannels,
                                                 desc->outChannels,
                                                 desc->dilationY,
                                                 desc->dilationX,
                                                 (float*)desc->weights.data());

  return swDesc;
}

/// Converts a BatchNormLayerDesc instance from C++ to Swift by creating a new SWBatchNormLayerDesc instance with the same properties.
/// - Parameter desc: The BatchNormLayerDesc instance to convert.
/// - Returns: A SWBatchNormLayerDesc instance with the same properties as the input BatchNormLayerDesc.
SWBatchNormLayerDesc MetalProcess::batchNormLayerDescToSwift(const BatchNormLayerDesc * desc) {
  vector<float> mergedScale = computeMergedScale(desc);
  vector<float> mergedBias = computeMergedBias(desc, mergedScale);
  float* mergedScalePtr = pinMetalTempFloatBuffer(std::move(mergedScale));
  float* mergedBiasPtr = pinMetalTempFloatBuffer(std::move(mergedBias));

  SWBatchNormLayerDesc swDesc =
  createSWBatchNormLayerDesc(desc->numChannels,
                             mergedScalePtr,
                             mergedBiasPtr);

  return swDesc;
}

/// Convert an activation layer description from C++ to Swift
/// - Parameter desc: An activation layer description
ActivationKind MetalProcess::activationLayerDescToSwift(const ActivationLayerDesc * desc) {

  switch (desc->activation) {
    case ACTIVATION_RELU:
      return ActivationKind::relu();
    case ACTIVATION_MISH:
      return ActivationKind::mish();
    case ACTIVATION_IDENTITY:
      return ActivationKind::identity();
    default:
      testAssert(false);
      return ActivationKind::identity(); // Placeholder for compilation
  }
}

/// Convert a residual block description from C++ to Swift
/// - Parameter desc: A residual block description
/// - Returns: The residual block description converted to SWResidualBlockDesc
SWResidualBlockDesc MetalProcess::residualBlockDescToSwift(const ResidualBlockDesc * desc) {

  SWBatchNormLayerDesc preBN = batchNormLayerDescToSwift(&desc->preBN);
  ActivationKind preActivationKind = activationLayerDescToSwift(&desc->preActivation);
  SWConvLayerDesc regularConv = convLayerDescToSwift(&desc->regularConv);
  SWBatchNormLayerDesc midBN = batchNormLayerDescToSwift(&desc->midBN);
  ActivationKind midActivationKind = activationLayerDescToSwift(&desc->midActivation);
  SWConvLayerDesc finalConv = convLayerDescToSwift(&desc->finalConv);

  SWResidualBlockDesc swDesc =
  createSWResidualBlockDesc(preBN,
                            preActivationKind,
                            regularConv,
                            midBN,
                            midActivationKind,
                            finalConv);

  return swDesc;
}

/// Convert a matrix multiplication layer description from C++ to Swift
/// - Parameter desc: A matrix multiplication layer description
/// - Returns: The matrix multiplication layer description converted to SWMatMulLayerDesc
SWMatMulLayerDesc MetalProcess::matMulLayerDescToSwift(const MatMulLayerDesc * desc) {

  SWMatMulLayerDesc swDesc = createSWMatMulLayerDesc(desc->inChannels,
                                                     desc->outChannels,
                                                     (float*)desc->weights.data());

  return swDesc;
}

/// Convert a global pooling residual block description from C++ to Swift
/// - Parameter desc: A global pooling residual block description
/// - Returns: The global pooling residual block description converted to SWGlobalPoolingResidualBlockDesc
SWGlobalPoolingResidualBlockDesc MetalProcess::globalPoolingResidualBlockDescToSwift(const GlobalPoolingResidualBlockDesc* desc) {

  SWBatchNormLayerDesc preBN = batchNormLayerDescToSwift(&desc->preBN);
  ActivationKind preActivationKind = activationLayerDescToSwift(&desc->preActivation);
  SWConvLayerDesc regularConv = convLayerDescToSwift(&desc->regularConv);
  SWConvLayerDesc gpoolConv = convLayerDescToSwift(&desc->gpoolConv);
  SWBatchNormLayerDesc gpoolBN = batchNormLayerDescToSwift(&desc->gpoolBN);
  ActivationKind gpoolActivationKind = activationLayerDescToSwift(&desc->gpoolActivation);
  SWMatMulLayerDesc gpoolToBiasMul = matMulLayerDescToSwift(&desc->gpoolToBiasMul);
  SWBatchNormLayerDesc midBN = batchNormLayerDescToSwift(&desc->midBN);
  ActivationKind midActivationKind = activationLayerDescToSwift(&desc->midActivation);
  SWConvLayerDesc finalConv = convLayerDescToSwift(&desc->finalConv);

  SWGlobalPoolingResidualBlockDesc swDesc =
  createSWGlobalPoolingResidualBlockDesc(preBN,
                                         preActivationKind,
                                         regularConv,
                                         gpoolConv,
                                         gpoolBN,
                                         gpoolActivationKind,
                                         gpoolToBiasMul,
                                         midBN,
                                         midActivationKind,
                                         finalConv);

  return swDesc;
}

/// Convert residual blocks from C++ to Swift
/// - Parameters:
///   - blocks: Residual blocks
///   - swBlocks: A pointer to an array of BlockDescriptor
swift::Array<BlockDescriptor> MetalProcess::residualBlocksToSwift(const vector<pair<int, unique_ptr_void>>& blocks) {

  auto builder = createBlockDescriptorBuilder();

  for (int i = 0; i < blocks.size(); i++) {

    void * blockDesc = blocks[i].second.get();

    if (blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      BlockDescriptor descriptor = globalPoolingResidualBlockDescToSwift((GlobalPoolingResidualBlockDesc*)blockDesc);
      builder.enque(descriptor);
    } else if (blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      BlockDescriptor descriptor = nestedBottleneckResidualBlockDescToSwift((NestedBottleneckResidualBlockDesc*)blockDesc);
      builder.enque(descriptor);
    } else {
      BlockDescriptor descriptor = residualBlockDescToSwift((ResidualBlockDesc*)blockDesc);
      builder.enque(descriptor);
    }
  }

  return builder.getBlockDescriptors();
}

/// Convert a nested bottleneck residual block description from C++ to Swift
/// - Parameter desc: A nested bottleneck residual block description
SWNestedBottleneckResidualBlockDesc MetalProcess::nestedBottleneckResidualBlockDescToSwift(const NestedBottleneckResidualBlockDesc* desc) {

  SWBatchNormLayerDesc preBN = batchNormLayerDescToSwift(&desc->preBN);
  ActivationKind preActivationKind = activationLayerDescToSwift(&desc->preActivation);
  SWConvLayerDesc preConv = convLayerDescToSwift(&desc->preConv);
  auto swBlocks = residualBlocksToSwift(desc->blocks);
  SWBatchNormLayerDesc postBN = batchNormLayerDescToSwift(&desc->postBN);
  ActivationKind postActivationKind = activationLayerDescToSwift(&desc->postActivation);
  SWConvLayerDesc postConv = convLayerDescToSwift(&desc->postConv);

  SWNestedBottleneckResidualBlockDesc swDesc =
  createSWNestedBottleneckResidualBlockDesc(preBN,
                                            preActivationKind,
                                            preConv,
                                            swBlocks,
                                            postBN,
                                            postActivationKind,
                                            postConv);

  return swDesc;
}

/// Gom2024 descriptors do not carry upstream SGF metadata encoder state.
swift::Optional<SWSGFMetadataEncoderDesc> MetalProcess::sGFMetadataEncoderDescToSwift() {
  return swift::Optional<SWSGFMetadataEncoderDesc>::none();
}

/// Convert a trunk description from C++ to Swift
/// - Parameter trunk: A trunk description
/// - Returns: The trunk description converted to SWTrunkDesc
SWTrunkDesc MetalProcess::trunkDescToSwift(const TrunkDesc * trunk) {

  SWConvLayerDesc initialConv = convLayerDescToSwift(&trunk->initialConv);
  SWMatMulLayerDesc initialMatMul = matMulLayerDescToSwift(&trunk->initialMatMul);
  auto sgfMetadataEncoder = sGFMetadataEncoderDescToSwift();
  auto swBlocks = residualBlocksToSwift(trunk->blocks);
  SWBatchNormLayerDesc trunkTipBN = batchNormLayerDescToSwift(&trunk->trunkTipBN);
  ActivationKind trunkTipActivation = activationLayerDescToSwift(&trunk->trunkTipActivation);

  SWTrunkDesc swTrunkDesc = createSWTrunkDesc(trunk->version,
                                              trunk->trunkNumChannels,
                                              trunk->midNumChannels,
                                              trunk->regularNumChannels,
                                              trunk->gpoolNumChannels,
                                              initialConv,
                                              initialMatMul,
                                              sgfMetadataEncoder,
                                              swBlocks,
                                              trunkTipBN,
                                              trunkTipActivation);

  return swTrunkDesc;
}

/// Convert a policy head description from C++ to Swift
/// - Parameter policyHead: A policy head description
/// - Returns: The policy head description converted to SWPolicyHeadDesc
SWPolicyHeadDesc MetalProcess::policyHeadDescToSwift(const PolicyHeadDesc * policyHead) {
  const int metalPolicyVersion = std::min(policyHead->version, kMetalFallbackPolicyVersion);

  SWConvLayerDesc p1Conv = convLayerDescToSwift(&policyHead->p1Conv);
  SWConvLayerDesc g1Conv = convLayerDescToSwift(&policyHead->g1Conv);
  SWBatchNormLayerDesc g1BN = batchNormLayerDescToSwift(&policyHead->g1BN);
  ActivationKind g1Activation = activationLayerDescToSwift(&policyHead->g1Activation);
  SWMatMulLayerDesc gpoolToBiasMul = matMulLayerDescToSwift(&policyHead->gpoolToBiasMul);
  SWBatchNormLayerDesc p1BN = batchNormLayerDescToSwift(&policyHead->p1BN);
  ActivationKind p1Activation = activationLayerDescToSwift(&policyHead->p1Activation);
  SWConvLayerDesc p2Conv = convLayerDescToSwift(&policyHead->p2Conv);
  SWMatMulLayerDesc gpoolToPassMul = matMulLayerDescToSwift(&policyHead->gpoolToPassMul);
  MatBiasLayerDesc fallbackPassBiasDesc = makeZeroMatBiasLayerDesc(policyHead->gpoolToPassMul.outChannels);
  SWMatBiasLayerDesc gpoolToPassBias =
      createSWMatBiasLayerDesc(fallbackPassBiasDesc.numChannels,
                               pinMetalTempFloatBuffer(std::move(fallbackPassBiasDesc.weights)));
  ActivationKind passActivation = ActivationKind::identity();
  SWMatMulLayerDesc gpoolToPassMul2 = matMulLayerDescToSwift(&policyHead->gpoolToPassMul);

  SWPolicyHeadDesc swPolicyHead = createSWPolicyHeadDesc(metalPolicyVersion,
                                                         p1Conv,
                                                         g1Conv,
                                                         g1BN,
                                                         g1Activation,
                                                         gpoolToBiasMul,
                                                         p1BN,
                                                         p1Activation,
                                                         p2Conv,
                                                         gpoolToPassMul,
                                                         gpoolToPassBias,
                                                         passActivation,
                                                         gpoolToPassMul2);

  return swPolicyHead;
}

/// Convert a matrix bias layer description from C++ to Swift
/// - Parameter desc: A matrix bias layer description
/// - Returns: The matrix bias layer description converted to SWMatBiasLayerDesc
SWMatBiasLayerDesc MetalProcess::matBiasLayerDescToSwift(const MatBiasLayerDesc * desc) {

  SWMatBiasLayerDesc swDesc = createSWMatBiasLayerDesc(desc->numChannels, (float*)desc->weights.data());

  return swDesc;
}

/// Convert a value head description from C++ to Swift
/// - Parameter valueHead: A value head description
/// - Returns: The value head description converted to SWValueHeadDesc
SWValueHeadDesc MetalProcess::valueHeadDescToSwift(const ValueHeadDesc * valueHead) {

  SWConvLayerDesc v1Conv = convLayerDescToSwift(&valueHead->v1Conv);
  SWBatchNormLayerDesc v1BN = batchNormLayerDescToSwift(&valueHead->v1BN);
  ActivationKind v1Activation = activationLayerDescToSwift(&valueHead->v1Activation);
  SWMatMulLayerDesc v2Mul = matMulLayerDescToSwift(&valueHead->v2Mul);
  SWMatBiasLayerDesc v2Bias = matBiasLayerDescToSwift(&valueHead->v2Bias);
  ActivationKind v2Activation = activationLayerDescToSwift(&valueHead->v2Activation);
  SWMatMulLayerDesc v3Mul = matMulLayerDescToSwift(&valueHead->v3Mul);
  SWMatBiasLayerDesc v3Bias = matBiasLayerDescToSwift(&valueHead->v3Bias);
  SWMatMulLayerDesc sv3Mul = matMulLayerDescToSwift(&valueHead->sv3Mul);
  SWMatBiasLayerDesc sv3Bias = matBiasLayerDescToSwift(&valueHead->sv3Bias);
  SWConvLayerDesc vOwnershipConv = convLayerDescToSwift(&valueHead->vOwnershipConv);

  SWValueHeadDesc swDesc = createSWValueHeadDesc(valueHead->version,
                                                 v1Conv,
                                                 v1BN,
                                                 v1Activation,
                                                 v2Mul,
                                                 v2Bias,
                                                 v2Activation,
                                                 v3Mul,
                                                 v3Bias,
                                                 sv3Mul,
                                                 sv3Bias,
                                                 vOwnershipConv);

  return swDesc;
}

SWModelDesc MetalProcess::modelDescToSwift(const ModelDesc* modelDesc) {
  return createSWModelDesc(modelDesc->version,
                           swift::String(modelDesc->name),
                           modelDesc->numInputChannels,
                           modelDesc->numInputGlobalChannels,
                           kMetalFallbackInputMetaChannels,
                           modelDesc->numValueChannels,
                           modelDesc->numScoreValueChannels,
                           modelDesc->numOwnershipChannels,
                           trunkDescToSwift(&modelDesc->trunk),
                           policyHeadDescToSwift(&modelDesc->policyHead),
                           valueHeadDescToSwift(&modelDesc->valueHead));
}

//---------------------------------------------------------------------------------------------------------

/**
 * @brief This function initializes the global state of the NeuralNet class upon program startup.
 * This function should be called only once upon program startup. It ensures that the global state
 * of the NeuralNet class is properly initialized, enabling it to function correctly throughout
 * the lifetime of the program.
 * Note that this function does not take any input parameters or return any values.
 */
void NeuralNet::globalInitialize() {
  // Do nothing.
}

/**
 * @brief This function cleans up the global state of the NeuralNet class at program termination.
 * This function should be called once at program termination. It ensures that the global state of
 * the NeuralNet class is properly cleaned up, freeing any resources that were allocated during the
 * lifetime of the program.
 * Note that this function does not take any input parameters or return any values.
 */
void NeuralNet::globalCleanup() {
  // Do nothing.
}

/**
 * @brief Loads a neural network model from a file.
 * This function creates a LoadedModel object by loading a neural network model from a file specified by
 * the `file` parameter and expected SHA-256 hash specified by the `expectedSha256` parameter. The LoadedModel
 * object is returned as a pointer.
 * @param file The name of the file containing the neural network model.
 * @param expectedSha256 The expected SHA-256 hash of the model file.
 * @return A pointer to the LoadedModel object created by loading the model file.
 */
LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file, expectedSha256);
  return loadedModel;
}

/**
 * @brief Frees memory used by a LoadedModel object.
 * This function deallocates memory used by a LoadedModel object specified by the `loadedModel` parameter.
 * @param loadedModel A pointer to the LoadedModel object to deallocate memory for.
 */
void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

string NeuralNet::getModelName(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.name;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

ComputeContext::ComputeContext(int nnX, int nnY, enabled_t useFP16Mode, enabled_t useNHWCMode):
metalComputeContext(createMetalComputeContext(nnX, nnY)) {
  this->useFP16Mode = useFP16Mode;

  SWEnable swUseFP16Mode =
  (useFP16Mode == enabled_t::False) ? SWEnable::False() :
  (useFP16Mode == enabled_t::True) ? SWEnable::True() :
  SWEnable::Auto();

  SWEnable swUseNHWCMode =
  (useNHWCMode == enabled_t::False) ? SWEnable::False() :
  (useNHWCMode == enabled_t::True) ? SWEnable::True() :
  SWEnable::Auto();
}

ComputeContext::~ComputeContext() {
}

/**
 * @brief Creates a ComputeContext object for computing neural network operations.
 * This function creates a ComputeContext object by setting configuration settings for neural network computations,
 * such as whether to use half-precision floating-point (FP16) mode and whether to use the NHWC format for input
 * tensors. The ComputeContext object is returned as a pointer.
 * @param gpuIdxs (Unused) A vector of GPU indices to use for computations.
 * @param logger (Unused) A pointer to a Logger object to use for logging messages.
 * @param nnXLen The width of the input tensor.
 * @param nnYLen The height of the input tensor.
 * @param openCLTunerFile (Unused) The name of a file containing OpenCL tuning parameters.
 * @param homeDataDirOverride (Unused) A directory to use for storing data.
 * @param openCLReTunePerBoardSize (Unused) Whether to re-tune OpenCL parameters for different board sizes.
 * @param useFP16Mode Whether to use half-precision floating-point (FP16) mode for computations.
 * @param useNHWCMode Whether to use the NHWC format for input tensors.
 * @param loadedModel (Unused) A pointer to a LoadedModel object containing a loaded neural network model.
 * @return A pointer to the ComputeContext object created.
 */
ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel) {

  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  return new ComputeContext(nnXLen, nnYLen, useFP16Mode, useNHWCMode);
}

/**
 * @brief Frees memory used by a ComputeContext object.
 * This function deallocates memory used by a ComputeContext object specified by the `computeContext` parameter.
 * @param computeContext A pointer to the ComputeContext object to deallocate memory for.
 */
void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------

ComputeHandle::ComputeHandle(ComputeContext* context,
                             const LoadedModel* loadedModel,
                             bool inputsUseNHWC,
                             int gpuIdx,
                             int serverThreadIdx):
metalhandle(maybeCreateMetalComputeHandle((gpuIdx < 100),
                                          serverThreadIdx,
                                          MetalProcess::modelDescToSwift(&loadedModel->modelDesc),
                                          context->metalComputeContext)) {

  const ModelDesc* modelDesc = &loadedModel->modelDesc;
  auto metalContext = context->metalComputeContext;

  nnXLen = metalContext.getNnXLen();
  nnYLen = metalContext.getNnYLen();
  gpuIndex = gpuIdx;
  version = modelDesc->version;
  metaEncoderVersion = 0;
  this->inputsUseNHWC = inputsUseNHWC;

  /* Use FP16 mode if the model supports it and the user has not explicitly
   * disabled it. */
  useFP16 = (context->useFP16Mode != enabled_t::False);

  (void)serverThreadIdx;
}

ComputeHandle::~ComputeHandle() {
}

static mutex computeHandleMutex;

/**
 * @brief Create a new ComputeHandle object for performing neural network computations.
 * This function creates a new ComputeHandle object for performing neural network computations,
 * using the specified parameters and settings. The object is allocated on the heap using the
 * 'new' operator and returned as a pointer.
 * @param context A pointer to the ComputeContext object to use for computation.
 * @param loadedModel A pointer to the LoadedModel object containing the neural network model to use.
 * @param logger A pointer to the Logger object to use for logging messages.
 * @param maxBatchSize The maximum batch size to use for computation.
 * @param requireExactNNLen Whether the neural network length must match the input data length exactly.
 * @param inputsUseNHWC Whether the input data uses NHWC format.
 * @param gpuIdxForThisThread The index of the GPU to use for computation.
 * @param serverThreadIdx The index of the server thread to use for computation.
 * @return A pointer to the newly-created ComputeHandle object.
 */
ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx) {

  (void)logger;
  (void)maxBatchSize;
  // Current implementation always tolerates excess nn len
  (void)requireExactNNLen;

  // Transfer the default GPU index into physical GPU index 0
  int gpuIdx = (gpuIdxForThisThread == -1) ? 0 : gpuIdxForThisThread;
  ComputeHandle* handle = nullptr;

  {
    lock_guard<mutex> lock(computeHandleMutex);
    clearMetalTempFloatBuffers();
    handle = new ComputeHandle(context, loadedModel, inputsUseNHWC, gpuIdx, serverThreadIdx);
    clearMetalTempFloatBuffers();
  }

  return handle;
}

/**
 * @brief Free the memory used by a ComputeHandle object.
 * This function frees the memory used by the specified ComputeHandle object, which was
 * previously allocated on the heap using the 'new' operator.
 * @param handle A pointer to the ComputeHandle object to free.
 */
void NeuralNet::freeComputeHandle(ComputeHandle* handle) {
  delete handle;
}

/**
 * @brief Check whether a ComputeHandle object is using 16-bit floating-point precision.
 * This function checks whether the specified ComputeHandle object is using 16-bit floating-point
 * precision for computation, and returns a boolean value indicating the result.
 * @param handle A pointer to the ComputeHandle object to check.
 * @return True if the ComputeHandle object is using 16-bit floating-point precision, false otherwise.
 */
bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return handle->useFP16;
}

//------------------------------------------------------------------------------

/**
 * @brief Print information about the available devices.
 */
void NeuralNet::printDevices() {
  printMetalDevices();
}

//--------------------------------------------------------------

/**
 * @brief Construct a new InputBuffers object for storing input data for neural network computation.
 * This constructor initializes a new InputBuffers object for storing input data for neural network
 * computation, based on the specified parameters and settings.
 * @param loadedModel A pointer to the LoadedModel object containing the neural network model to use.
 * @param maxBatchSz The maximum batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 */
InputBuffers::InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
  const ModelDesc& m = loadedModel->modelDesc;

  maxBatchSize = maxBatchSz;
  // Gom2024 still uses a single board policy plane plus a separate pass logit,
  // matching the existing OpenCL/Eigen backends rather than upstream Go metal.
  policyResultChannels = 1;

  singleSpatialElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
  singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
  singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
  singleInputMetaElts = (size_t)kMetalFallbackInputMetaChannels;
  singlePolicyResultElts = (size_t)(nnXLen * nnYLen);
  singlePolicyPassResultElts = 1;
  singlePolicyProbsElts = (size_t)((nnXLen * nnYLen) + 1);
  singleValueResultElts = (size_t)m.numValueChannels;
  singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
  singleOwnerMapElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
  singleScoreValuesResultElts = (size_t)m.numScoreValueChannels;

  assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
  assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);
  assert(singleValueResultElts == 3);

  rowSpatialBufferElts = (size_t)maxBatchSz * singleSpatialElts;
  userInputBufferElts = (size_t)maxBatchSize * singleInputElts;
  userInputGlobalBufferElts = (size_t)maxBatchSize * singleInputGlobalElts;
  userInputMetaBufferElts = (size_t)maxBatchSize * singleInputMetaElts;
  policyResultBufferElts = (size_t)maxBatchSize * singlePolicyResultElts * policyResultChannels;
  policyPassResultBufferElts = (size_t)maxBatchSize * singlePolicyPassResultElts * policyResultChannels;
  policyProbsBufferElts = (size_t)maxBatchSize * singlePolicyProbsElts * policyResultChannels;
  valueResultBufferElts = (size_t)maxBatchSize * singleValueResultElts;
  ownershipResultBufferElts = (size_t)maxBatchSize * singleOwnershipResultElts;
  ownerMapBufferElts = (size_t)maxBatchSz * singleOwnerMapElts;
  scoreValuesResultBufferElts = (size_t)maxBatchSize * singleScoreValuesResultElts;

  rowSpatialBuffer = new float[rowSpatialBufferElts];
  userInputBuffer = new float[userInputBufferElts];
  // Zero out the input buffer for arbitrary board sizes
  memset(&userInputBuffer[0], 0, userInputBufferElts * sizeof(userInputBuffer[0]));

  userInputGlobalBuffer = new float[userInputGlobalBufferElts];
  userInputMetaBuffer = new float[userInputMetaBufferElts];
  policyResults = new float[policyResultBufferElts];
  policyPassResults = new float[policyPassResultBufferElts];
  policyProbsBuffer = new float[policyProbsBufferElts];
  valueResults = new float[valueResultBufferElts];
  ownershipResults = new float[ownershipResultBufferElts];
  ownerMapBuffer = new float[ownerMapBufferElts];
  scoreValuesResults = new float[scoreValuesResultBufferElts];
}

/**
 * @brief Destroy the InputBuffers object and free all associated memory.
 * This destructor destroys the InputBuffers object and frees all memory associated with it,
 * including all input and output buffers used for neural network computation.
 */
InputBuffers::~InputBuffers() {
  delete[] rowSpatialBuffer;
  delete[] userInputBuffer;
  delete[] userInputGlobalBuffer;
  delete[] userInputMetaBuffer;
  delete[] policyResults;
  delete[] policyPassResults;
  delete[] policyProbsBuffer;
  delete[] valueResults;
  delete[] ownershipResults;
  delete[] ownerMapBuffer;
  delete[] scoreValuesResults;
}

/**
 * @brief Create a new InputBuffers object for storing input data for neural network computation.
 * This function creates a new InputBuffers object for storing input data for neural network computation,
 * using the specified parameters and settings. The object is allocated on the heap using the 'new' operator
 * and returned as a pointer.
 * @param loadedModel A pointer to the LoadedModel object containing the neural network model to use.
 * @param maxBatchSize The maximum batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 * @return A pointer to the newly-created InputBuffers object.
 */
InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

/**
 * @brief Free the memory used by an InputBuffers object.
 * This function frees the memory used by the specified InputBuffers object, which was
 * previously allocated on the heap using the 'new' operator.
 * @param inputBuffers A pointer to the InputBuffers object to free.
 */
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//--------------------------------------------------------------

void MetalProcess::copyRowData(float* dest, const float* src, size_t numElements) {
  copy(src, src + numElements, dest);
}

/**
 * @brief Convert input data from NHWC format to NCHW format in-place if necessary.
 *
 * @param rowSpatialInput Pointer to the input data (single batch element assumed).
 * @param C Number of channels.
 * @param H Height.
 * @param W Width.
 * @param inputsUseNHWC Flag indicating if the input data is currently in NHWC format.
 */
void MetalProcess::convertNCHW(
    float* rowSpatialInput,
    const int C,
    const int H,
    const int W,
    const bool inputsUseNHWC) {

  if ((!inputsUseNHWC) || (C <= 0) || (H <= 0) || (W <= 0)) {
    return;
  }

  const int totalSize = H * W * C;

  if (totalSize <= 1)
    return;

  const int HW = H * W;

  auto get_nchw_target_index = [C, W, HW](int nhwc_index) -> int {
    int c = nhwc_index % C;
    int temp = nhwc_index / C;
    int x = temp % W;
    int y = temp / W;
    return (c * HW) + (y * W) + x;
  };

  std::vector<bool> processed(totalSize, false);

  for (int i = 0; i < totalSize; ++i) {
    if (processed[i])
      continue;

    int target_i = get_nchw_target_index(i);

    if (target_i == i) {
      processed[i] = true;
      continue;
    }

    int current_idx = i;
    float value_in_hand = rowSpatialInput[i];

    while (true) {
      int target_idx = get_nchw_target_index(current_idx);
      float value_at_target = rowSpatialInput[target_idx];
      rowSpatialInput[target_idx] = value_in_hand;
      processed[target_idx] = true;
      value_in_hand = value_at_target;
      current_idx = target_idx;

      if (current_idx == i)
        break;
    }
  }
}

void MetalProcess::processRowData(size_t row, ComputeHandle* gpuHandle, InputBuffers* inputBuffers, NNResultBuf** inputBufs) {
  int nnXLen = gpuHandle->nnXLen;
  int nnYLen = gpuHandle->nnYLen;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(gpuHandle->version);

  float* rowSpatialInput = &inputBuffers->userInputBuffer[inputBuffers->singleSpatialElts * row];
  float* rowGlobalInput = &inputBuffers->userInputGlobalBuffer[inputBuffers->singleInputGlobalElts * row];
  float* rowMetaInput = &inputBuffers->userInputMetaBuffer[inputBuffers->singleInputMetaElts * row];
  const float* rowGlobal = inputBufs[row]->rowGlobal;
  const float* rowSpatial = inputBufs[row]->rowSpatial;

  MetalProcess::copyRowData(rowGlobalInput, rowGlobal, inputBuffers->singleInputGlobalElts);
  for(size_t i = 0; i < inputBuffers->singleInputMetaElts; i++)
    rowMetaInput[i] = 0.0f;

  SymmetryHelpers::copyInputsWithSymmetry(
    rowSpatial,
    rowSpatialInput,
    1,
    nnYLen,
    nnXLen,
    numSpatialFeatures,
    gpuHandle->inputsUseNHWC,
    inputBufs[row]->symmetry);

  MetalProcess::convertNCHW(
    rowSpatialInput,
    numSpatialFeatures,
    nnYLen,
    nnXLen,
    gpuHandle->inputsUseNHWC);
}

void MetalProcess::processPolicy(
  InputBuffers* inputBuffers,
  const ComputeHandle* gpuHandle,
  NNResultBuf* inputBuf,
  size_t row,
  float* outputPolicys) {
  auto& buffers = *inputBuffers;
  float* targetBuffer = &buffers.policyResults[row * buffers.singlePolicyResultElts];
  const auto symmetry = inputBuf->symmetry;
  float* policyProbs = outputPolicys + row * NNPos::MAX_NN_POLICY_SIZE;

  SymmetryHelpers::copyOutputsWithSymmetry(
    targetBuffer, policyProbs, 1, gpuHandle->nnYLen, gpuHandle->nnXLen, symmetry);
  policyProbs[buffers.singlePolicyResultElts] = buffers.policyPassResults[row];
}

void MetalProcess::processValue(
  const InputBuffers* inputBuffers,
  NNOutput* currentOutput,
  const size_t row) {
  const size_t singleValueResultElts = inputBuffers->singleValueResultElts;
  assert(singleValueResultElts == 3);
  const float* valueOutputBuf = &inputBuffers->valueResults[row * singleValueResultElts];
  currentOutput->whiteWinProb = valueOutputBuf[0];
  currentOutput->whiteLossProb = valueOutputBuf[1];
  currentOutput->whiteNoResultProb = valueOutputBuf[2];
}

void MetalProcess::processScoreValues(
  const InputBuffers* inputBuffers,
  NNOutput* currentOutput,
  const int modelVersion,
  const size_t row) {
  const size_t offset = row * inputBuffers->singleScoreValuesResultElts;
  const float* currentScoreValueData = &inputBuffers->scoreValuesResults[offset];

  if(modelVersion >= 9) {
    size_t numScoreValueChannels = inputBuffers->singleScoreValuesResultElts;
    assert(numScoreValueChannels == 6);
    currentOutput->varTimeLeft = currentScoreValueData[3];
    currentOutput->shorttermWinlossError = currentScoreValueData[4];
  }
  else if(modelVersion >= 8) {
    size_t numScoreValueChannels = inputBuffers->singleScoreValuesResultElts;
    assert(numScoreValueChannels == 4);
    currentOutput->varTimeLeft = currentScoreValueData[3];
    currentOutput->shorttermWinlossError = 0;
  }
  else if(modelVersion >= 4) {
    size_t numScoreValueChannels = inputBuffers->singleScoreValuesResultElts;
    assert(numScoreValueChannels == 2);
    currentOutput->varTimeLeft = 0;
    currentOutput->shorttermWinlossError = 0;
  }
  else {
    assert(modelVersion >= 3);
    currentOutput->varTimeLeft = 0;
    currentOutput->shorttermWinlossError = 0;
  }
}

void MetalProcess::processRow(
  size_t row,
  const ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs,
  float* outputPolicys) {
  NNOutput* currentOutput = outputs[row];
  assert(currentOutput->nnXLen == gpuHandle->nnXLen);
  assert(currentOutput->nnYLen == gpuHandle->nnYLen);
  MetalProcess::processPolicy(inputBuffers, gpuHandle, inputBufs[row], row, outputPolicys);
  MetalProcess::processValue(inputBuffers, currentOutput, row);
  MetalProcess::processScoreValues(inputBuffers, currentOutput, gpuHandle->version, row);
}

/**
 * @brief Compute the neural network output using Metal API and the specified input data and GPU handle.
 * This function computes the neural network output using the Metal API and the specified input data and ComputeHandle
 * object for GPU acceleration. The computed output is stored in the specified vector of NNOutput pointers.
 * @param gpuHandle A pointer to the ComputeHandle object to use for GPU computation.
 * @param inputBuffers A pointer to the InputBuffers object containing the input data for computation.
 * @param numBatchEltsFilled The number of batch elements filled in the input buffer.
 * @param inputBufs An array of pointers to NNResultBuf objects containing the neural network input data.
 * @param outputs A vector of NNOutput pointers to store the computed output.
 */
void MetalProcess::getMetalOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs,
  float* outputPolicys) {
  assert(numBatchEltsFilled > 0);

  int batchSize = numBatchEltsFilled;

  assert(batchSize <= inputBuffers->maxBatchSize);
  assert((NNModelVersion::getNumSpatialFeatures(gpuHandle->version) * gpuHandle->nnXLen * gpuHandle->nnYLen) <= inputBuffers->singleInputElts);
  assert(NNModelVersion::getNumGlobalFeatures(gpuHandle->version) == inputBuffers->singleInputGlobalElts);

  assert(inputBuffers->singleValueResultElts == 3);

  for(size_t row = 0; row < batchSize; row++) {
    MetalProcess::processRowData(row, gpuHandle, inputBuffers, inputBufs);
  }

  auto metalHandle = gpuHandle->metalhandle;
  assert(metalHandle);

  metalHandle.get().apply(inputBuffers->userInputBuffer,
                          inputBuffers->userInputGlobalBuffer,
                          inputBuffers->userInputMetaBuffer,
                          inputBuffers->policyResults,
                          inputBuffers->policyPassResults,
                          inputBuffers->valueResults,
                          inputBuffers->scoreValuesResults,
                          inputBuffers->ownershipResults,
                          batchSize);

  for(size_t row = 0; row < batchSize; row++) {
    MetalProcess::processRow(row, gpuHandle, inputBuffers, inputBufs, outputs, outputPolicys);
  }
}

/**
 * @brief Compute the neural network output using the specified input data and GPU handle.
 * This function computes the neural network output using the specified input data and ComputeHandle object
 * for GPU acceleration. The computed output is stored in the specified vector of NNOutput pointers.
 * @param gpuHandle A pointer to the ComputeHandle object to use for GPU computation.
 * @param inputBuffers A pointer to the InputBuffers object containing the input data for computation.
 * @param numBatchEltsFilled The number of batch elements filled in the input buffer.
 * @param inputBufs An array of pointers to NNResultBuf objects containing the neural network input data.
 * @param outputs A vector of NNOutput pointers to store the computed output.
 */
void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs,
  float* outputPolicys) {

  MetalProcess::getMetalOutput(gpuHandle, inputBuffers, numBatchEltsFilled, inputBufs, outputs, outputPolicys);
}

bool MetalProcess::testEvaluateConv(const ConvLayerDesc* desc,
                                    int batchSize,
                                    int nnXLen,
                                    int nnYLen,
                                    const vector<float>& inputBuffer,
                                    vector<float>& outputBuffer) {

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->outChannels;
  outputBuffer.resize(numOutputFloats);

  testConvLayer(convLayerDescToSwift(desc),
                nnXLen,
                nnYLen,
                batchSize,
                (float*)inputBuffer.data(),
                (float*)outputBuffer.data());

  return true;
}

/**
 * @brief Evaluate a convolutional layer using Metal API for testing purposes.
 * This function evaluates a convolutional layer using the Metal API for testing purposes.
 * The input buffer and output buffer are specified as vectors of floats, and the result of the computation
 * is stored in the output buffer. The function returns true if the evaluation is implemented.
 * @param desc A pointer to the ConvLayerDesc object describing the convolutional layer to evaluate.
 * @param batchSize The batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 * @param useFP16 A boolean indicating whether to use half-precision floating point format for computation.
 * @param useNHWC A boolean indicating whether to use NHWC layout for input and output buffers.
 * @param inputBuffer A vector of floats containing the input buffer data.
 * @param outputBuffer A vector of floats to store the computed output.
 * @return true if the convolutional layer evaluation is implemented, false otherwise.
 */
bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer) {

  (void)useFP16;
  (void)useNHWC;
  return MetalProcess::testEvaluateConv(desc, batchSize, nnXLen, nnYLen, inputBuffer, outputBuffer);
}

bool MetalProcess::testEvaluateBatchNorm(const BatchNormLayerDesc* desc,
                                         int batchSize,
                                         int nnXLen,
                                         int nnYLen,
                                         const vector<float>& inputBuffer,
                                         const vector<float>& maskBuffer,
                                         vector<float>& outputBuffer) {

  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->numChannels;
  outputBuffer.resize(numOutputFloats);

  testBatchNormLayer(batchNormLayerDescToSwift(desc),
                     nnXLen,
                     nnYLen,
                     batchSize,
                     (float*)inputBuffer.data(),
                     (float*)maskBuffer.data(),
                     (float*)outputBuffer.data());

  return true;
}

/**
 * @brief Evaluate a batch normalization layer using Metal API for testing purposes.
 * This function evaluates a batch normalization layer using the Metal API for testing purposes.
 * The input buffer and output buffer are specified as vectors of floats, and the result of the computation
 * is stored in the output buffer. The function returns true if the evaluation is implemented.
 * @param desc A pointer to the BatchNormLayerDesc object describing the batch normalization layer to evaluate.
 * @param batchSize The batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 * @param useFP16 A boolean indicating whether to use half-precision floating point format for computation.
 * @param useNHWC A boolean indicating whether to use NHWC layout for input and output buffers.
 * @param inputBuffer A vector of floats containing the input buffer data.
 * @param maskBuffer A vector of floats containing the mask buffer data. Mask should be in 'NHW' format (no "C" channel).
 * @param outputBuffer A vector of floats to store the computed output.
 * @return true if the batch normalization layer evaluation is implemented, false otherwise.
 */
bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {

  (void)useFP16;
  (void)useNHWC;
  return MetalProcess::testEvaluateBatchNorm(desc, batchSize, nnXLen, nnYLen, inputBuffer, maskBuffer, outputBuffer);
}

bool MetalProcess::testEvaluateResidualBlock(const ResidualBlockDesc* desc,
                                             int batchSize,
                                             int nnXLen,
                                             int nnYLen,
                                             const vector<float>& inputBuffer,
                                             const vector<float>& maskBuffer,
                                             vector<float>& outputBuffer) {

  size_t numTrunkFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  outputBuffer.resize(numTrunkFloats);

  testResidualBlock(residualBlockDescToSwift(desc),
                    batchSize,
                    nnXLen,
                    nnYLen,
                    (float*)inputBuffer.data(),
                    (float*)maskBuffer.data(),
                    (float*)outputBuffer.data());

  return true;
}

/**
 * @brief Evaluate a residual block using Metal API for testing purposes.
 * This function evaluates a residual block using the Metal API for testing purposes.
 * The input buffer and output buffer are specified as vectors of floats, and the result of the computation
 * is stored in the output buffer. The function returns true if the evaluation is implemented.
 * @param desc A pointer to the ResidualBlockDesc object describing the residual block to evaluate.
 * @param batchSize The batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 * @param useFP16 A boolean indicating whether to use half-precision floating point format for computation.
 * @param useNHWC A boolean indicating whether to use NHWC layout for input and output buffers.
 * @param inputBuffer A vector of floats containing the input buffer data.
 * @param maskBuffer A vector of floats containing the mask buffer data.
 * @param outputBuffer A vector of floats to store the computed output.
 * @return true if the residual block evaluation is implemented, false otherwise.
 */
bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {

  (void)useFP16;
  (void)useNHWC;
  return MetalProcess::testEvaluateResidualBlock(desc, batchSize, nnXLen, nnYLen, inputBuffer, maskBuffer, outputBuffer);
}

bool MetalProcess::testEvaluateGlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc* desc,
                                                          int batchSize,
                                                          int nnXLen,
                                                          int nnYLen,
                                                          const vector<float>& inputBuffer,
                                                          const vector<float>& maskBuffer,
                                                          vector<float>& outputBuffer) {

  size_t numTrunkFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  outputBuffer.resize(numTrunkFloats);

  testGlobalPoolingResidualBlock(globalPoolingResidualBlockDescToSwift(desc),
                                 batchSize,
                                 nnXLen,
                                 nnYLen,
                                 (float*)inputBuffer.data(),
                                 (float*)maskBuffer.data(),
                                 (float*)outputBuffer.data());

  return true;
}

/**
 * @brief Evaluate a global pooling residual block using Metal API for testing purposes.
 * This function evaluates a global pooling residual block using the Metal API for testing purposes.
 * The input buffer and output buffer are specified as vectors of floats, and the result of the computation
 * is stored in the output buffer. The function returns true if the evaluation is implemented.
 * @param desc A pointer to the GlobalPoolingResidualBlockDesc object describing the global pooling residual block to
 * evaluate.
 * @param batchSize The batch size to use for computation.
 * @param nnXLen The x length of the neural network computation context.
 * @param nnYLen The y length of the neural network computation context.
 * @param useFP16 A boolean indicating whether to use half-precision floating point format for computation.
 * @param useNHWC A boolean indicating whether to use NHWC layout for input and output buffers.
 * @param inputBuffer A vector of floats containing the input buffer data.
 * @param maskBuffer A vector of floats containing the mask buffer data.
 * @param outputBuffer A vector of floats to store the computed output.
 * @return true if the global pooling residual block evaluation is implemented, false otherwise.
 */
bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {

  (void)useFP16;
  (void)useNHWC;
  return MetalProcess::testEvaluateGlobalPoolingResidualBlock(desc, batchSize, nnXLen, nnYLen, inputBuffer, maskBuffer, outputBuffer);
}

#endif  // USE_METAL_BACKEND
