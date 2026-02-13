#include <bits/stdc++.h>

using namespace std;

/**
 * PROJECT: OMNIVERSE C++ TEMPLATE
 * AUTHOR: AMIRMOHAMMAD GHASEMZADEH
 * YEAR: 2026
 * TARGET: ADVANCED COMPETITIVE PROGRAMMING (QUERA/ICPC)
 */

// --- OPTIMIZATION CORE (The Beast Mode) ---
#pragma GCC optimize("Ofast,unroll-loops,inline,no-stack-protector")
#pragma GCC target("avx,avx2,fma,bmi,bmi2,popcnt,lzcnt")

// --- FAST IO & MACROS ---
#define fastio() ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL)
#define pb push_back
#define eb emplace_back
#define mp make_pair
#define fi first
#define se second
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define sz(x) ((int)(x).size())
#define mem(a, b) memset(a, b, sizeof(a))
#define watch(x) cerr << #x << " = " << x << endl

// --- TYPE DEFINITIONS ---
typedef long long ll;
typedef __int128_t int128; // Massive numbers up to 10^38
typedef long double ld;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<int> vi;
typedef vector<ll> vll;

// --- CONSTANTS ---
const int MOD = 1e9 + 7;
const int MOD2 = 998244353;
const ll INF = 2e18;
const ld EPS = 1e-12;
const ld PI = acos(-1.0);

// ============================================================================
// 1. THE MATH GALAXY (Number Theory & Combinatorics)
// ============================================================================
namespace Math {
    ll power(ll b, ll e, ll m = MOD) {
        ll r = 1; b %= m;
        while (e > 0) { if (e & 1) r = (int128)r * b % m; b = (int128)b * b % m; e >>= 1; }
        return r;
    }
    ll modInverse(ll n, ll m = MOD) { return power(n, m - 2, m); }
    
    // Fast GCD (CPU Level)
    ll gcd(ll a, ll b) { return __gcd(a, b); }
    ll lcm(ll a, ll b) { return (a / gcd(a, b)) * b; }

    // Sieve of Eratosthenes (Up to 10^7)
    const int MAXP = 10000001;
    bitset<MAXP> is_prime;
    void sieve() {
        is_prime.set(); is_prime[0] = is_prime[1] = 0;
        for (int p = 2; p * p < MAXP; p++)
            if (is_prime[p]) for (int i = p * p; i < MAXP; i += p) is_prime[i] = 0;
    }

    // Matrix Exponentiation (O(dim^3 log exp))
    struct Matrix {
        static const int DIM = 2; // Change based on problem
        ll mat[DIM][DIM];
        Matrix() { mem(mat, 0); }
        static Matrix identity() {
            Matrix res; for(int i=0; i<DIM; i++) res.mat[i][i] = 1;
            return res;
        }
        Matrix operator*(const Matrix& other) const {
            Matrix res;
            for(int i=0; i<DIM; i++)
                for(int k=0; k<DIM; k++)
                    for(int j=0; j<DIM; j++)
                        res.mat[i][j] = (res.mat[i][j] + mat[i][k] * other.mat[k][j]) % MOD;
            return res;
        }
    };
    Matrix matPow(Matrix a, ll b) {
        Matrix res = Matrix::identity();
        while(b > 0) { if(b & 1) res = res * a; a = a * a; b >>= 1; }
        return res;
    }
}

// ============================================================================
// 2. THE GRAPH GALAXY (DSU, Dijkstra, Trees)
// ============================================================================
namespace Graph {
    struct DSU {
        vi p, s;
        DSU(int n) { p.resize(n+1); s.assign(n+1, 1); iota(all(p), 0); }
        int find(int i) { return (p[i] == i) ? i : (p[i] = find(p[i])); }
        void unite(int i, int j) {
            int ri = find(i), rj = find(j);
            if (ri != rj) {
                if (s[ri] < s[rj]) swap(ri, rj);
                p[rj] = ri; s[ri] += s[rj];
            }
        }
    };

    // Dijkstra (O(E log V))
    vll dijkstra(int start, int n, vector<pii> adj[]) {
        vll dist(n + 1, INF);
        priority_queue<pll, vector<pll>, greater<pll>> pq;
        dist[start] = 0; pq.push({0, start});
        while (!pq.empty()) {
            ll d = pq.top().first; int u = pq.top().second; pq.pop();
            if (d > dist[u]) continue;
            for (auto& edge : adj[u]) {
                if (dist[u] + edge.second < dist[edge.first]) {
                    dist[edge.first] = dist[u] + edge.second;
                    pq.push({dist[edge.first], edge.first});
                }
            }
        }
        return dist;
    }
}

// ============================================================================
// 3. THE DATA STRUCTURE GALAXY (Segment Tree, Fenwick)
// ============================================================================
namespace DS {
    struct Fenwick { // Binary Indexed Tree (O(log N))
        int n; vi tree;
        Fenwick(int n) : n(n), tree(n + 1, 0) {}
        void update(int i, int delta) { for (; i <= n; i += i & -i) tree[i] += delta; }
        int query(int i) { int sum = 0; for (; i > 0; i -= i & -i) sum += tree[i]; return sum; }
    };

    struct SegmentTree { // Point Update, Range Query
        int n; vi tree;
        SegmentTree(int n) : n(n), tree(4 * n, 0) {}
        void update(int node, int start, int end, int idx, int val) {
            if(start == end) { tree[node] = val; return; }
            int mid = (start + end) / 2;
            if(idx <= mid) update(2*node, start, mid, idx, val);
            else update(2*node+1, mid+1, end, idx, val);
            tree[node] = tree[2*node] + tree[2*node+1]; // Change for max/min
        }
        int query(int node, int start, int end, int l, int r) {
            if(r < start || end < l) return 0;
            if(l <= start && end <= r) return tree[node];
            int mid = (start + end) / 2;
            return query(2*node, start, mid, l, r) + query(2*node+1, mid+1, end, l, r);
        }
    };
}

// ============================================================================
// 4. THE GEOMETRY GALAXY
// ============================================================================
namespace Geo {
    struct Point {
        ld x, y;
        Point operator-(const Point& o) const { return {x - o.x, y - o.y}; }
        ld cross(const Point& o) const { return x * o.y - y * o.x; }
        ld dist(const Point& o) const { return hypot(x - o.x, y - o.y); }
    };
    ld polygonArea(vector<Point>& p) {
        ld area = 0;
        for (int i = 0; i < sz(p); i++) area += p[i].cross(p[(i + 1) % sz(p)]);
        return abs(area) / 2.0;
    }
}

// ============================================================================
// 5. THE STRING GALAXY
// ============================================================================
namespace Str {
    // KMP String Matching (O(N+M))
    vi computeLPS(string pat) {
        int m = sz(pat); vi lps(m);
        for (int i = 1, j = 0; i < m; i++) {
            while (j > 0 && pat[i] != pat[j]) j = lps[j - 1];
            if (pat[i] == pat[j]) j++;
            lps[i] = j;
        }
        return lps;
    }
}

// ============================================================================
// CORE SOLVER (The Place to Code)
// ============================================================================
void solve() {
    // 1. Inputs
    // 2. Choose Galaxy (Namespace)
    // 3. Logic
    // 4. Output
}

int main() {
    fastio();
    cout << fixed << setprecision(12);
    
    // For Problems with multiple Test Cases:
    /*
    int t; if(!(cin >> t)) return 0;
    while(t--) solve();
    */

    solve(); // Single test case
    return 0;
}

/**
 * ?? THE ULTIMATE SURVIVAL CHEAT SHEET (Simple Language)
 * ----------------------------------------------------------------------------
 * 1. DSU (Group Manager): 
 * - Use 'Graph::DSU dsu(n);'
 * - 'dsu.unite(1, 2);' connects person 1 and 2.
 * - 'dsu.find(1) == dsu.find(2)' checks if they are in the same group.
 *
 * 2. BINPOW (Fast Power):
 * - Use 'Math::power(base, exp);' for (base^exp) % MOD.
 * - Essential when exp is very large (up to 10^18).
 *
 * 3. INT128 (Big Numbers):
 * - Use 'int128 x;' for numbers bigger than 10^18.
 * - You can read/write them using 'cin >> x;' and 'cout << x;'.
 *
 * 4. FENWICK (Quick Sums):
 * - Use 'DS::Fenwick ft(n);'
 * - 'ft.add(idx, val);' to change a value.
 * - 'ft.query(idx);' to get sum from start to idx in O(log N).
 *
 * 5. TRICKS:
 * - Use '1LL * a * b' to prevent overflow in long long multiplication.
 * - 'abs(a - b) < EPS' to compare floating point numbers.
 * - Use '\n' instead of 'endl' to avoid TLE (Time Limit Exceeded).
 */