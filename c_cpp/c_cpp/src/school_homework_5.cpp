#include "iostream"
#include <array>
#include <cmath>
#include <utility>
double distence(std::pair<double, double> p1, std::pair<double, double> p2) {
  double dx = p1.first - p2.first;
  double dy = p1.second - p2.second;
  return std::sqrt(dx * dx + dy * dy);
}
class Shape {
public:
  virtual double printArea() = 0;
};
class Rectangle : public Shape {
private:
  std::pair<std::pair<double, double>, std::pair<double, double>> points;

public:
  Rectangle(double x1, double y1, double x2, double y2)
      : points({{x1, y1}, {x2, y2}}){};
  double printArea() {
    double width = std::abs(points.first.second - points.second.second);
    double height = std::abs(points.first.first - points.second.first);
    return width * height;
  }
};
class Circle : public Shape {
private:
  double radius;
  std::pair<double, double> dot;

public:
  Circle(double r, double x, double y) : radius(r), dot({x, y}){};
  double printArea() { return M_PI * std::pow(radius, 2); }
};
class Triangle : public Shape {
private:
  std::array<std::pair<double, double>, 3> points;

public:
  Triangle(double x1, double y1, double x2, double y2, double x3, double y3) {
    points[0] = {x1, y1};
    points[1] = {x2, y2};
    points[2] = {x3, y3};
  }
  double printArea() {
    double s = 0.5 * distence(points[0], points[1]) +
               0.5 * distence(points[1], points[2]) +
               0.5 * distence(points[2], points[0]);
    return std::sqrt(s *
                     (s - distence(points[0], points[1]) *
                              (s - distence(points[1], points[2]) *
                                       (s - distence(points[2], points[0])))));
  }
};
int main() {
  Rectangle r(0, 0, 10, 10);
  Circle c(5, 5, 5);
  Triangle t(0, 0, 10, 0, 5, 10);
  std::cout << "Rectangle area: " << r.printArea() << std::endl;
  std::cout << "Circle area: " << c.printArea() << std::endl;
  std::cout << "Triangle area: " << t.printArea() << std::endl;
  return 0;
}