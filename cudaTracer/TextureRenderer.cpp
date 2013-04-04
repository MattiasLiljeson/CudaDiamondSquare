#include "TextureRenderer.h"
#include "Vertex.h"
#include "LayoutFactory.h"

//=========================================================================
// Public functions
//=========================================================================
TextureRenderer::TextureRenderer( DeviceHandler* p_deviceHandler, int p_texWidth, int p_texHeight )
{
	m_deviceHandler = p_deviceHandler;
	m_texWidth = p_texWidth;
	m_texHeight = p_texHeight;
	m_shaderSet = NULL;
	m_inputLayout = NULL,

	initTexture();
	initShaders();
	initInputLayout();
	initQuad();
	initStates();
	initInterop();
}


TextureRenderer::~TextureRenderer()
{
	delete m_shaderSet;
	m_shaderSet = NULL;

	SAFE_RELEASE( m_inputLayout );
}

void TextureRenderer::update( float p_dt )
{
	cudaStream_t    stream = 0;
	const int nbResources = 1;
	cudaGraphicsResource *ppResources[nbResources] =
	{
		m_textureSet.cudaResource,
	};
	cudaGraphicsMapResources(nbResources, ppResources, stream);
	getLastCudaError("cudaGraphicsMapResources(3) failed");

	static float t = 0.0f;
	// populate the 2d texture
	{
		cudaArray *cuArray;
		cudaGraphicsSubResourceGetMappedArray(&cuArray, m_textureSet.cudaResource, 0, 0);
		cudaError_t err = cudaGetLastError();

		// kick off the kernel and send the staging buffer
		// cudaLinearMemory as an argument to allow the kernel to
		// write to it
		cuda_texture_2d(m_textureSet.cudaLinearMemory,
			m_textureSet.width, m_textureSet.height, m_textureSet.pitch, t);
		getLastCudaError("cuda_texture_2d failed");

		// then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
		cudaMemcpy2DToArray(
			cuArray, // dst array
			0, 0,    // offset
			m_textureSet.cudaLinearMemory, m_textureSet.pitch,       // src
			m_textureSet.width*4*sizeof(float), m_textureSet.height, // extent
			cudaMemcpyDeviceToDevice); // kind
		getLastCudaError("cudaMemcpy2DToArray failed");
	}

	cudaGraphicsUnmapResources(nbResources, ppResources, stream);
	getLastCudaError("cudaGraphicsUnmapResources(3) failed");
}

void TextureRenderer::draw()
{
	m_deviceHandler->getContext()->VSSetShader( m_shaderSet->m_vs, NULL, 0 );
	m_deviceHandler->getContext()->PSSetShader( m_shaderSet->m_ps, NULL, 0 );
	m_deviceHandler->getContext()->IASetInputLayout( m_inputLayout );
	m_deviceHandler->getContext()->IASetPrimitiveTopology(
		D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );

	const int SLOT = 0;
	const int BUFFER_CNT = 1;
	const unsigned int STRIDE[] = {sizeof(Vertex)};
	const unsigned int OFFSET[] = {0};
	m_deviceHandler->getContext()->IASetVertexBuffers(
		SLOT, BUFFER_CNT, &m_vertexBuffer, STRIDE, OFFSET );

	const int VERTEX_CNT = 6;
	const int START_VERTEX = 0;
	m_deviceHandler->getContext()->Draw( VERTEX_CNT, START_VERTEX );

	m_deviceHandler->presentFrame();
}

//=========================================================================
// Private functions
//=========================================================================
void TextureRenderer::initTexture()
{
	ID3D11Device* device = m_deviceHandler->getDevice();
	ID3D11DeviceContext* context = m_deviceHandler->getContext();

	m_textureSet.width  = m_texWidth;
	m_textureSet.height = m_texHeight;

	D3D11_TEXTURE2D_DESC desc;
	ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
	desc.Width = m_textureSet.width;
	desc.Height = m_textureSet.height;
	desc.MipLevels = 1;
	desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

	HR(device->CreateTexture2D(&desc, NULL, &m_textureSet.pTexture));

	HR(device->CreateShaderResourceView(m_textureSet.pTexture, NULL, &m_textureSet.pSRView));

	m_textureSet.offsetInShader = 0; // to be clean we should look for the offset from the shader code
	context->PSSetShaderResources(m_textureSet.offsetInShader, 1, &m_textureSet.pSRView);
}

void TextureRenderer::initShaders()
{
	m_shaderSet = new ShaderSet( m_deviceHandler );
	m_shaderSet->createSet( "../shader.hlsl", "VS", "PS" );
}

void TextureRenderer::initInputLayout()
{
	LayoutDesc desc = LayoutFactory::getPointTexCoordDesc();
	HRESULT hr = m_deviceHandler->getDevice()->CreateInputLayout(
		desc.m_layoutPtr,
		desc.m_elementCnt,
		m_shaderSet->m_vsData->GetBufferPointer(),
		m_shaderSet->m_vsData->GetBufferSize(),
		&m_inputLayout );
}

void TextureRenderer::initQuad()
{
	Vertex mesh[]= {
		{{ 1,	-1,	0},	{ 1, 1}},
		{{ -1,	-1,	0},	{ 0, 1}},
		{{ 1,	1,	0},	{ 1, 0}},

		{{ -1, -1,	0},	{ 0, 1}},
		{{ 1,	1,	0},	{ 1, 0}},
		{{ -1,	1,	0},	{ 0, 0}}
	};

	D3D11_BUFFER_DESC bd;
	bd.ByteWidth = sizeof(mesh);
	bd.Usage = D3D11_USAGE_IMMUTABLE;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;

	D3D11_SUBRESOURCE_DATA vertBuffSubRes;
	vertBuffSubRes.pSysMem = &mesh[0];

	HR(m_deviceHandler->getDevice()->CreateBuffer(&bd, &vertBuffSubRes, &m_vertexBuffer));

	//Buffer<PTVertex>* quadBuffer;

	//// Create description for buffer
	//BufferConfig::BUFFER_INIT_DESC bufferDesc;
	//bufferDesc.ElementSize = sizeof(PTVertex);
	//bufferDesc.Usage = BufferConfig::BUFFER_DEFAULT;
	//bufferDesc.NumElements = 6;
	//bufferDesc.Type = BufferConfig::VERTEX_BUFFER;
	//bufferDesc.Slot = BufferConfig::SLOT0;

	//// Create buffer from config and data
	//quadBuffer = new Buffer<PTVertex>(m_device,m_deviceContext,&mesh[0],bufferDesc);

	//return quadBuffer;
}

void TextureRenderer::initStates()
{
	D3D11_RASTERIZER_DESC rasterizerDesc;
	rasterizerDesc.FillMode = D3D11_FILL_SOLID;
	rasterizerDesc.CullMode = D3D11_CULL_NONE;
	rasterizerDesc.FrontCounterClockwise = FALSE;
	rasterizerDesc.DepthClipEnable = FALSE;
	rasterizerDesc.ScissorEnable = FALSE;
	rasterizerDesc.AntialiasedLineEnable = FALSE;
	rasterizerDesc.MultisampleEnable = FALSE;
	rasterizerDesc.DepthBias = 0;
	rasterizerDesc.DepthBiasClamp = 0.0f;
	rasterizerDesc.SlopeScaledDepthBias = 0.0f;

	m_deviceHandler->getDevice()->CreateRasterizerState(&rasterizerDesc, &m_rsDefault);

	// set the changed values for wireframe mode
	rasterizerDesc.FillMode = D3D11_FILL_WIREFRAME;
	rasterizerDesc.CullMode = D3D11_CULL_NONE;
	rasterizerDesc.AntialiasedLineEnable = TRUE;

	m_deviceHandler->getDevice()->CreateRasterizerState(&rasterizerDesc, &m_rsWireframe);

	if(false)
		m_deviceHandler->getContext()->RSSetState(m_rsWireframe);
	else
		m_deviceHandler->getContext()->RSSetState(m_rsDefault);
}

void TextureRenderer::initInterop()
{
	// begin interop
	cudaD3D11SetDirect3DDevice(m_deviceHandler->getDevice());
	getLastCudaError("cudaD3D11SetDirect3DDevice failed");

	// 2D
	// register the Direct3D resources that we'll use
	// we'll read to and write from m_textureSet, so don't set any special
	// map flags for it
	cudaGraphicsD3D11RegisterResource(&m_textureSet.cudaResource,
		m_textureSet.pTexture, cudaGraphicsRegisterFlagsNone);
	getLastCudaError("cudaGraphicsD3D11RegisterResource (m_textureSet) failed");

	// cuda cannot write into the texture directly : the texture is seen
	// as a cudaArray and can only be mapped as a texture
	// Create a buffer so that cuda can write into it
	// pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
	cudaMallocPitch(&m_textureSet.cudaLinearMemory, &m_textureSet.pitch,
		m_textureSet.width * sizeof(float) * 4, m_textureSet.height);
	getLastCudaError("cudaMallocPitch (m_textureSet) failed");
	cudaMemset(m_textureSet.cudaLinearMemory, 1,
		m_textureSet.pitch * m_textureSet.height);
}