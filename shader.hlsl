
Texture2D g_tex;

struct VertexIn
{
	float3 position : POSITION;
	float2 texCoord : TEXCOORD;
};

struct VertexOut
{
    float4 position : SV_POSITION;
	float2 texCoord : TEXCOORD;
};

VertexOut VS( VertexIn p_input )
{
	VertexOut vout;
	vout.position = float4(p_input.position,1.0f);
	vout.texCoord = p_input.texCoord;
    
	return vout;
}

float4 PS( VertexOut p_input ) : SV_TARGET
{
	uint3 idx = uint3( p_input.position.x, p_input.position.y, 0 );
	return g_tex.Load(idx);
	return float4( p_input.texCoord.x, p_input.texCoord.y, 0.3f, 1.0f );
}