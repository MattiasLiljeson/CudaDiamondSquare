#include "ShaderSet.h"

#include <windows.h>
#include "Utils.h"
#include <D3Dcompiler.h>

ShaderSet::ShaderSet( DeviceHandler* p_deviceHandler )
{
	m_deviceHandler = p_deviceHandler;

	m_vsData = NULL;
	m_vs = NULL;
	m_psData = NULL;
	m_ps = NULL;
}


ShaderSet::~ShaderSet()
{
	SAFE_RELEASE( m_vsData );
	SAFE_RELEASE( m_vs );
	SAFE_RELEASE( m_psData );
	SAFE_RELEASE( m_ps );
}

void ShaderSet::createSet( string p_filePath, string p_vsEntry, string p_psEntry )
{
	compileShader( p_filePath, p_vsEntry, "vs_5_0", &m_vsData );
	if( m_vsData != NULL) {
		createVs();
	}

	compileShader( p_filePath, p_psEntry, "ps_5_0", &m_psData );
	if( m_psData != NULL)  {
		createPs();
	}
}

void ShaderSet::createVs()
{
	HR(m_deviceHandler->getDevice()->CreateVertexShader(
		m_vsData->GetBufferPointer(),
		m_vsData->GetBufferSize(),
		NULL, &m_vs));
}

void ShaderSet::createPs()
{
	HR(m_deviceHandler->getDevice()->CreatePixelShader(
		m_psData->GetBufferPointer(),
		m_psData->GetBufferSize(),
		NULL, &m_ps));
}

void ShaderSet::compileShader( 
	const string &p_sourceFile, const string &p_entryPoint,
	const string &p_profile, ID3DBlob** out_blob )
{
	ID3DBlob*	compilationErrors = NULL;
	ID3DBlob*	shaderBlob = NULL;

	*out_blob = NULL;

	DWORD compileFlags = D3DCOMPILE_ENABLE_STRICTNESS;

#if defined(DEBUG) || defined(_DEBUG)
	compileFlags |= D3DCOMPILE_DEBUG;
	compileFlags |= D3DCOMPILE_SKIP_OPTIMIZATION;
	//compileFlags |= D3DCOMPILE_WARNINGS_ARE_ERRORS;
#else
	compileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif

	wstring wsSource = L"";
	Utils::wstringFromString( wsSource, p_sourceFile );

	HRESULT res = S_OK;
	D3DX11CompileFromFile(
		wsSource.c_str(),
		NULL,
		NULL,
		p_entryPoint.c_str(), p_profile.c_str(),
		compileFlags, 0,
		NULL,
		&shaderBlob, &compilationErrors,
		&res);

	HRESULT hrCopy = res;
	if(FAILED(res))
	{
		if( compilationErrors )
		{
			MessageBoxA(0, (char*)compilationErrors->GetBufferPointer(), 0, 0);
			SAFE_RELEASE(compilationErrors);
		}
		else
		{
			HR(hrCopy);
		}
	}

	*out_blob = shaderBlob;
}