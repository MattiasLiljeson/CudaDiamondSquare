#ifndef UTILS_H
#define UTILS_H

#include <string>

#include <d3dx10.h>
#include <dxerr.h>
#include <cassert>

using namespace std;

//*************************************************************************
// Simple d3d error checker for book demos.
//*************************************************************************

#if defined(DEBUG) | defined(_DEBUG)
	#ifndef HR
	#define HR(x)                                              \
	{                                                          \
		HRESULT hr = (x);                                      \
		if(FAILED(hr))                                         \
		{                                                      \
			DXTrace(__FILE__, (DWORD)__LINE__, hr, L#x, true); \
		}                                                      \
	}
	#endif

#else
	#ifndef HR
	#define HR(x) (x)
	#endif
#endif

// Release COM objects if not NULL and set them to NULL
#define SAFE_RELEASE(x)											\
	if( x )														\
	{															\
		x->Release();											\
		(x) = NULL; 											\
	}

//wstring wstrFromString(string from)
//{
//	wstring to(from.length(), L' ');
//	copy(from.begin(), from.end(), to.begin());
//	return to;
//}

class Utils
{
public:
	static void wstringFromString(std::wstring &ws, const std::string &s)
	{
		std::wstring wsTmp(s.begin(), s.end());
		ws = wsTmp;
	}

	static void stringFromWstring(const std::wstring &ws, std::string &s)
	{
		std::string sTmp(ws.begin(), ws.end());
		s = sTmp;
	}
};

struct IdxOutOfRange : std::exception {
	const char* what() const throw() {return "Index out of range\n";}
};

struct VecIdx
{
	enum Idx 
	{
		Idx_NA = -1, Idx_FIRST,

		X = Idx_FIRST, Y, Z, W,

		Idx_LAST = W, Idx_CNT
	};
};

struct Vec2 
{
	int x;
	int y;

	int& operator[]( unsigned int idx )
	{
		switch( idx )
		{
		case VecIdx::X:
			return x;
		case VecIdx::Y:
			return y;
		default:
			IdxOutOfRange e;
			throw e;
		}
		return x; // THIS SHOULD NEVER HAPPEN!
	}

	const int& operator[](int idx) const
	{

	}
};

#endif //UTILS_H