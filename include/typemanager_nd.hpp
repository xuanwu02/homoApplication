#ifndef _SZ_TYPEMANAGER_ND_2_HPP
#define _SZ_TYPEMANAGER_ND_2_HPP

#include <stdio.h>
#include <string.h>
#include <cstddef>
#include <cstdlib>
#include "typemanager.hpp"

inline void convertByte2Int_1b_2D(
    size_t          intArrayLength,   
    unsigned char*  byteArray,
    size_t          byteArrayLength,
    int*            intArray,
	unsigned char*  signFlag,
    bool            applySignHere,
    int             size_x,
    int             size_y,
    size_t          dim0_offset
){
    size_t i = 0;
    size_t n = 0;
    for (; i < byteArrayLength - 1 && n + 8 <= intArrayLength; i++)
    {
        unsigned char tmp = byteArray[i];
        for (int b = 0; b < 8; b++)
        {
            int shift = 7 - b;
            int bit   = (tmp >> shift) & 0x01;

            int row = static_cast<int>(n / size_y);
            int col = static_cast<int>(n % size_y);
            intArray[row * dim0_offset + col] = bit;
            n++;
        }
    }
    if (i < byteArrayLength && n < intArrayLength)
    {
        unsigned char tmp = byteArray[i];
        size_t bitsRemaining = intArrayLength - n;
        if (bitsRemaining > 8) bitsRemaining = 8;
        for (size_t b = 0; b < bitsRemaining; b++)
        {
            int shift = 7 - static_cast<int>(b);
            int bit   = (tmp >> shift) & 0x01;
            int row = static_cast<int>(n / size_y);
            int col = static_cast<int>(n % size_y);
            intArray[row * dim0_offset + col] = bit;
            n++;
        }
    }
}

inline void convertByte2Int_2b_2D(
    size_t          stepLength,
    unsigned char*  byteArray,
    size_t          byteArrayLength,
    int*            intArray,
	unsigned char*  signFlag,
    bool            applySignHere,
    int             size_x,
    int             size_y,
    size_t          dim0_offset
){
    const size_t totalElems = size_x * size_y;
    size_t n = 0;
    size_t i = 0;
    int mod4 = static_cast<int>(stepLength % 4);
    if (mod4 == 0)
    {
        for (i = 0; i < byteArrayLength && (n + 4) <= totalElems; i++)
        {
            unsigned char tmp = byteArray[i];
            int v0 = (tmp & 0xC0) >> 6;
            int v1 = (tmp & 0x30) >> 4;
            int v2 = (tmp & 0x0C) >> 2;
            int v3 =  tmp & 0x03;
            {
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = v0;
                n++;
            }
            {
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = v1;
                n++;
            }
            {
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = v2;
                n++;
            }
            {
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = v3;
                n++;
            }
        }
    }
    else
    {
        size_t t = (byteArrayLength > 0) ? (byteArrayLength - 1) : 0;
        for (i = 0; i < t && (n + 4) <= totalElems; i++)
        {
            unsigned char tmp = byteArray[i];
            int v0 = (tmp & 0xC0) >> 6;
            int v1 = (tmp & 0x30) >> 4;
            int v2 = (tmp & 0x0C) >> 2;
            int v3 =  tmp & 0x03;
            {
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = v0;
                n++;
            }
            {
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = v1;
                n++;
            }
            {
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = v2;
                n++;
            }
            {
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = v3;
                n++;
            }
        }
        if (i < byteArrayLength && n < totalElems)
        {
            unsigned char tmp = byteArray[i];
            switch (mod4)
            {
            case 1:
            {
                int v0 = (tmp & 0xC0) >> 6;
                if (n < totalElems)
                {
                    size_t row = n / size_y;
                    size_t col = n % size_y;
                    intArray[row * dim0_offset + col] = v0;
                    n++;
                }
                break;
            }
            case 2:
            {
                int v0 = (tmp & 0xC0) >> 6;
                int v1 = (tmp & 0x30) >> 4;
                if (n < totalElems)
                {
                    size_t row = n / size_y;
                    size_t col = n % size_y;
                    intArray[row * dim0_offset + col] = v0;
                    n++;
                }
                if (n < totalElems)
                {
                    size_t row = n / size_y;
                    size_t col = n % size_y;
                    intArray[row * dim0_offset + col] = v1;
                    n++;
                }
                break;
            }
            case 3:
            {
                int v0 = (tmp & 0xC0) >> 6;
                int v1 = (tmp & 0x30) >> 4;
                int v2 = (tmp & 0x0C) >> 2;
                if (n < totalElems)
                {
                    size_t row = n / size_y;
                    size_t col = n % size_y;
                    intArray[row * dim0_offset + col] = v0;
                    n++;
                }
                if (n < totalElems)
                {
                    size_t row = n / size_y;
                    size_t col = n % size_y;
                    intArray[row * dim0_offset + col] = v1;
                    n++;
                }
                if (n < totalElems)
                {
                    size_t row = n / size_y;
                    size_t col = n % size_y;
                    intArray[row * dim0_offset + col] = v2;
                    n++;
                }
                break;
            }
            }
        }
    }
}

inline void convertByte2Int_3b_2D(
    size_t          stepLength,
    unsigned char*  byteArray,
    size_t          byteArrayLength,
    int*            intArray,
	unsigned char*  signFlag,
    bool            applySignHere,
    int             size_x,
    int             size_y,
    size_t          dim0_offset
){
    const size_t totalElems = size_x * size_y;
    if (byteArrayLength == 0 || totalElems == 0)
        return;

    size_t i  = 0;
    size_t n  = 0;
    unsigned char tmp = byteArray[i];
    while (n < stepLength && n < totalElems)
    {
        switch (n % 8)
        {
        case 0:
            {
                int val = (tmp & 0xE0) >> 5;
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = val;
                n++;
            }
            break;

        case 1:
            {
                int val = (tmp & 0x1C) >> 2;
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = val;
                n++;
            }
            break;

        case 2:
            {
                unsigned int ii = (tmp & 0x03) << 1;
                i++;
                if (i >= byteArrayLength) return;
                tmp = byteArray[i];
                ii |= (tmp & 0x80) >> 7;

                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = static_cast<int>(ii);
                n++;
            }
            break;

        case 3:
            {
                int val = (tmp & 0x70) >> 4;
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = val;
                n++;
            }
            break;

        case 4:
            {
                int val = (tmp & 0x0E) >> 1;
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = val;
                n++;
            }
            break;

        case 5:
            {
                unsigned int ii = (tmp & 0x01) << 2;
                i++;
                if (i >= byteArrayLength) return;
                tmp = byteArray[i];
                ii |= (tmp & 0xC0) >> 6;

                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = static_cast<int>(ii);
                n++;
            }
            break;

        case 6:
            {
                int val = (tmp & 0x38) >> 3;
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = val;
                n++;
            }
            break;

        case 7:
            {
                int val = (tmp & 0x07);
                size_t row = n / size_y;
                size_t col = n % size_y;
                intArray[row * dim0_offset + col] = val;
                n++;
                i++;
                if (i >= byteArrayLength) return;
                tmp = byteArray[i];
            }
            break;
        }
    }
}

inline void convertByte2Int_4b_2D(
    size_t          stepLength,
    unsigned char*  byteArray,
    size_t          byteArrayLength,
    int*            intArray,
	unsigned char*  signFlag,
    bool            applySignHere,
    int             size_x,
    int             size_y,
    size_t          dim0_offset
){
    const size_t totalElems = size_x * size_y;
    const size_t maxToWrite = (stepLength < totalElems) ? stepLength : totalElems;
    size_t i = 0;
    size_t n = 0;
    for (; i < byteArrayLength && n < maxToWrite; i++)
    {
        unsigned char tmp = byteArray[i];
        {
            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = (tmp & 0xF0) >> 4;
            n++;
            if (n >= maxToWrite) break;
        }
        {
            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = tmp & 0x0F;
            n++;
        }
    }
}

inline void convertByte2Int_5b_2D(
    size_t          stepLength,
    unsigned char*  byteArray,
    size_t          byteArrayLength,
    int*            intArray,
	unsigned char*  signFlag,
    bool            applySignHere,
    int             size_x,
    int             size_y,
    size_t          dim0_offset
){
    const size_t totalElems = size_x * size_y;
    if (byteArrayLength == 0 || totalElems == 0) {
        return;
    }

    size_t i = 0;
    size_t n = 0;
    unsigned char tmp = byteArray[i];
    while (n < stepLength && n < totalElems)
    {
        switch (n % 8)
        {
        case 0:
        {
            int val = (tmp & 0xF8) >> 3;
            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = val;
            n++;
            break;
        }
        case 1:
        {
            unsigned int ii = (tmp & 0x07) << 2;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0xC0) >> 6;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 2:
        {
            int val = (tmp & 0x3E) >> 1;
            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = val;
            n++;
            break;
        }
        case 3:
        {
            unsigned int ii = (tmp & 0x01) << 4;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0xF0) >> 4;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 4:
        {
            unsigned int ii = (tmp & 0x0F) << 1;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0x80) >> 7;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 5:
        {
            int val = (tmp & 0x7C) >> 2;
            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = val;
            n++;
            break;
        }
        case 6:
        {
            unsigned int ii = (tmp & 0x03) << 3;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0xE0) >> 5;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 7:
        {
            int val = (tmp & 0x1F);
            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = val;
            n++;

            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            break;
        }
        }
    }
}

inline void convertByte2Int_6b_2D(
    size_t          stepLength,
    unsigned char*  byteArray,
    size_t          byteArrayLength,
    int*            intArray,
	unsigned char*  signFlag,
    bool            applySignHere,
    int             size_x,
    int             size_y,
    size_t          dim0_offset
){
    const size_t totalElems = size_x * size_y;
    if (byteArrayLength == 0 || totalElems == 0) {
        return;
    }

    size_t i = 0;
    size_t n = 0;
    unsigned char tmp = byteArray[i];
    while (n < stepLength && n < totalElems)
    {
        switch (n % 4)
        {
        case 0:
        {
            int val = (tmp & 0xFC) >> 2;
            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = val;
            n++;
            break;
        }
        case 1:
        {
            unsigned int ii = (tmp & 0x03) << 4;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0xF0) >> 4;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 2:
        {
            unsigned int ii = (tmp & 0x0F) << 2;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0xC0) >> 6;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 3:
        {
            int val = (tmp & 0x3F);
            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = val;
            n++;

            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            break;
        }
        }
    }
}

inline void convertByte2Int_7b_2D(
    size_t          stepLength,
    unsigned char*  byteArray,
    size_t          byteArrayLength,
    int*            intArray,
	unsigned char*  signFlag,
    bool            applySignHere,
    int             size_x,
    int             size_y,
    size_t          dim0_offset
){
    const size_t totalElems = size_x * size_y;
    if (byteArrayLength == 0 || totalElems == 0) {
        return;
    }
    size_t i = 0;
    size_t n = 0;
    unsigned char tmp = byteArray[i];
    while (n < stepLength && n < totalElems)
    {
        switch (n % 8)
        {
        case 0:
        {
            int val = (tmp & 0xFE) >> 1;
            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = val;
            n++;
            break;
        }
        case 1:
        {
            unsigned int ii = (tmp & 0x01) << 6;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0xFC) >> 2;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 2:
        {
            unsigned int ii = (tmp & 0x03) << 5;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0xF8) >> 3;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 3:
        {
            unsigned int ii = (tmp & 0x07) << 4;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0xF0) >> 4;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 4:
        {
            unsigned int ii = (tmp & 0x0F) << 3;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0xE0) >> 5;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 5:
        {
            unsigned int ii = (tmp & 0x1F) << 2;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0xC0) >> 6;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 6:
        {
            unsigned int ii = (tmp & 0x3F) << 1;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            ii |= (tmp & 0x80) >> 7;

            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = static_cast<int>(ii);
            n++;
            break;
        }
        case 7:
        {
            int val = (tmp & 0x7F);
            size_t row = n / size_y;
            size_t col = n % size_y;
            intArray[row * dim0_offset + col] = val;
            n++;
            i++;
            if (i >= byteArrayLength) return;
            tmp = byteArray[i];
            break;
        }
        }
    }
}

size_t extract_fixed_length_bits_2D(
	unsigned char*  result,
	size_t 		    intArrayLength,
	int*		    signintArray,
	unsigned int    bit_count,
	unsigned char*  signFlag,
	int 		    size_x,
	int 		    size_y,
	size_t 		    dim0_offset
){
	unsigned int byte_count = bit_count / 8;
	unsigned int remainder_bit = bit_count % 8;
	size_t byteLength = byte_count * intArrayLength + (remainder_bit * intArrayLength - 1) / 8 + 1;
	size_t byte_offset = byte_count * intArrayLength;
	if(remainder_bit == 0) byteLength = byte_offset;
	if(remainder_bit > 0){
        bool applySignHere = byte_count == 0;
		switch (remainder_bit)
		{
		case 1:
			convertByte2Int_1b_2D(intArrayLength, result + byte_offset, (intArrayLength - 1) / 8 + 1, signintArray, signFlag, applySignHere, size_x, size_y, dim0_offset);
			break;
		case 2:
			convertByte2Int_2b_2D(intArrayLength, result + byte_offset, (intArrayLength * 2 - 1) / 8 + 1, signintArray, signFlag, applySignHere, size_x, size_y, dim0_offset);
			break;
		case 3:
			convertByte2Int_3b_2D(intArrayLength, result + byte_offset, (intArrayLength * 3 - 1) / 8 + 1, signintArray, signFlag, applySignHere, size_x, size_y, dim0_offset);
			break;
		case 4:
			convertByte2Int_4b_2D(intArrayLength, result + byte_offset, (intArrayLength * 4 - 1) / 8 + 1, signintArray, signFlag, applySignHere, size_x, size_y, dim0_offset);
			break;
		case 5:
			convertByte2Int_5b_2D(intArrayLength, result + byte_offset, (intArrayLength * 5 - 1) / 8 + 1, signintArray, signFlag, applySignHere, size_x, size_y, dim0_offset);
			break;
		case 6:
			convertByte2Int_6b_2D(intArrayLength, result + byte_offset, (intArrayLength * 6 - 1) / 8 + 1, signintArray, signFlag, applySignHere, size_x, size_y, dim0_offset);
			break;
		case 7:
			convertByte2Int_7b_2D(intArrayLength, result + byte_offset, (intArrayLength * 7 - 1) / 8 + 1, signintArray, signFlag, applySignHere, size_x, size_y, dim0_offset);
			break;
		default:
			printf("Error: try to extract %d bits\n", remainder_bit);
		}
	}

	size_t i, j;
	size_t n = 0;
    if (byte_count > 0)
    {
        if (remainder_bit == 0)
        {
            for (int k = 0; k < size_x; k++)
            {
                memset(signintArray + k * dim0_offset, 0, size_y * sizeof(int));
            }
        }
        size_t totalElements = intArrayLength;
        i = 0;
        j = 0;
        while (i < totalElements && n < byteLength)
        {
			unsigned int tmp1 = 0;
			unsigned int tmp2 = 0;
			for (unsigned c = 0; c < byte_count && n < byteLength; c++)
			{
				tmp1 = result[n];
				tmp1 <<= (8 * c);
				tmp2 |= tmp1;
				n++;
			}
			tmp2 <<= remainder_bit;
			size_t row = i / size_y;
			size_t col = i % size_y;
			signintArray[row * dim0_offset + col] |= tmp2;
			if (signFlag[i])
			{
				signintArray[row * dim0_offset + col] = -(signintArray[row * dim0_offset + col]);
			}
            i++;
        }
    }
    else
    {
        for (size_t i = 0; i < intArrayLength; i++)
        {
            size_t row = i / size_y;
            size_t col = i % size_y;

            if (signFlag[i] == 1)
            {
                signintArray[row * dim0_offset + col] = - signintArray[row * dim0_offset + col];
            }
        }
    }

	return byteLength;
}


#endif